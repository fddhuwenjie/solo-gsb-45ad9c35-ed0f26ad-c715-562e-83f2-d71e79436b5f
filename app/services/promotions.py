"""晋级名单：冻结排名裁决的发布与席位放弃/候补提升。

- publish_promotion_list：在「所有已生成轮次结束且无待处理赛果复核」
  时发布一份晋级名单。请求携带 request_id 幂等键、expected_event_seq
  乐观快照、资格轮次（须为已生成的最后一轮）与晋级名额 N；在同一写
  事务内重新计算当时的排名积分与 Buchholz，按排名积分降序、Buchholz
  降序、参赛者编号升序固定裁决，冻结前 N 名为晋级席位、其余参赛者
  按排名冻结为候补顺序，写入 promotion_list.published 事件。发布后
  原排名与扣分变化不改写该版本；再次发布生成新版本，旧版本原样保留。
- forfeit_seat：对已冻结名单的席位标记放弃。只有该席位当前有效晋级
  者可操作；在同一事务内把最前候补（position 最小的 waiting 候补）
  提升为该席位，写入 promotion_list.substitute_promoted 事件；无候补
  时席位保持空缺。重复 request_id 原样返回首次结果。
- 幂等：request_id 在赛事内唯一（发布与放弃共用）。重复请求参数一致
  时原样返回首次响应快照（不产生任何写入与事件）；同键不同参数 409。
- 查询：按赛事列出版本、按版本查询详情（每席位原始与当前参赛者、
  状态与冻结依据，候补顺序与状态）。

注意：本模块的写函数只在调用方事务内写入并 flush，不提交；名单行、
席位/候补变更、放弃记录、幂等记录与 promotion_list.* 事件由路由层在
同一事务中统一提交。所有调用均处于进程级 writer_lock 内，并由
(tournament_id, request_id) 唯一约束在数据库层兜底。
"""
from fastapi import HTTPException
from sqlalchemy import func
from sqlalchemy.orm import Session

from ..events.service import (
    PROMOTION_LIST_PUBLISHED,
    PROMOTION_LIST_SUBSTITUTE_PROMOTED,
    append_event,
)
from ..models import (
    Event,
    Participant,
    PromotionAction,
    PromotionForfeit,
    PromotionList,
    PromotionSeat,
    PromotionSubstitute,
    Round,
)
from ..schemas import (
    PromotionBasis,
    PromotionForfeitRequest,
    PromotionForfeitResponse,
    PromotionForfeitView,
    PromotionHolder,
    PromotionListPublishRequest,
    PromotionListResponse,
    PromotionParticipantRef,
    PromotionSeatView,
    PromotionSubstituteView,
)
from .announcements import current_event_seq
from .result_reviews import has_pending_review_for_tournament
from .scoring import Record, compute_records, half_to_float

PUBLISH = "publish"
FORFEIT = "forfeit"
ACTIVE = "active"
VACANT = "vacant"
WAITING = "waiting"
PROMOTED = "promoted"
FORFEITED = "forfeited"


# ---------- 幂等记录 ----------

def _find_replay(
    db: Session, tournament_id: int, request_id: str, action: str, fingerprint: dict
) -> dict | None:
    """同 request_id 的已成功请求：动作与参数一致返回首次响应，否则 409。"""
    row = (
        db.query(PromotionAction)
        .filter(
            PromotionAction.tournament_id == tournament_id,
            PromotionAction.request_id == request_id,
        )
        .first()
    )
    if row is None:
        return None
    if row.action == action and row.params == fingerprint:
        return row.response
    raise HTTPException(
        status_code=409,
        detail="request_id 已被用于该赛事的其他晋级名单请求，请更换 request_id",
    )


def _save_action(
    db: Session,
    tournament_id: int,
    list_id: int,
    request_id: str,
    action: str,
    fingerprint: dict,
    response: dict,
) -> None:
    db.add(
        PromotionAction(
            tournament_id=tournament_id,
            list_id=list_id,
            request_id=request_id,
            action=action,
            params=fingerprint,
            response=response,
        )
    )
    db.flush()


def _check_event_seq_snapshot(
    db: Session, tournament_id: int, expected_event_seq: int
) -> None:
    """先完成全部资格校验，再在同一写事务内确认事件序号快照未变化。"""
    actual_seq = current_event_seq(db, tournament_id)
    if expected_event_seq != actual_seq:
        raise HTTPException(
            status_code=409,
            detail=(
                f"赛事快照已变化：当前最大事件序号为 {actual_seq}，"
                f"请求期望 {expected_event_seq}"
            ),
        )


# ---------- 序列化 ----------

def _basis_snapshot(record: Record, rank: int) -> dict:
    """发布时冻结的排名依据（半分单位换算为对外分值）。"""
    return {
        "rank": rank,
        "points": half_to_float(record.points),
        "penalty": half_to_float(record.penalty),
        "rank_points": half_to_float(record.rank_points),
        "buchholz": half_to_float(record.buchholz),
    }


def _basis_of(
    participants: dict[int, Participant], participant_id: int, snapshot: dict
) -> PromotionBasis:
    p = participants[participant_id]
    return PromotionBasis(
        participant_id=participant_id,
        number=p.number,
        name=p.name,
        rank=snapshot["rank"],
        points=snapshot["points"],
        penalty=snapshot["penalty"],
        rank_points=snapshot["rank_points"],
        buchholz=snapshot["buchholz"],
    )


def serialize_list(db: Session, promotion_list: PromotionList) -> PromotionListResponse:
    """组装名单详情：每席位原始与当前参赛者、状态、冻结依据与放弃记录。"""
    participants = {
        p.id: p
        for p in db.query(Participant)
        .filter(Participant.tournament_id == promotion_list.tournament_id)
        .all()
    }
    seats = list(
        db.query(PromotionSeat)
        .filter(PromotionSeat.list_id == promotion_list.id)
        .order_by(PromotionSeat.seat_no)
    )
    substitutes = list(
        db.query(PromotionSubstitute)
        .filter(PromotionSubstitute.list_id == promotion_list.id)
        .order_by(PromotionSubstitute.position)
    )
    forfeits = list(
        db.query(PromotionForfeit)
        .filter(PromotionForfeit.list_id == promotion_list.id)
        .order_by(PromotionForfeit.id)
    )

    substitutes_by_participant = {s.participant_id: s for s in substitutes}
    forfeits_by_seat: dict[int, list[PromotionForfeit]] = {}
    for f in forfeits:
        forfeits_by_seat.setdefault(f.seat_id, []).append(f)
    seat_no_by_id = {s.id: s.seat_no for s in seats}

    seat_views: list[PromotionSeatView] = []
    for seat in seats:
        original = _basis_of(
            participants, seat.original_participant_id, seat.original_basis
        )
        current: PromotionHolder | None = None
        if seat.current_participant_id is not None:
            if seat.current_participant_id == seat.original_participant_id:
                current = PromotionHolder(
                    **original.model_dump(), source="original"
                )
            else:
                sub = substitutes_by_participant[seat.current_participant_id]
                # 每位候补至多被提升一次：提升它的放弃记录唯一
                promoting_forfeit = next(
                    f
                    for f in forfeits_by_seat.get(seat.id, [])
                    if f.promoted_substitute_id == sub.id
                )
                current = PromotionHolder(
                    **_basis_of(participants, sub.participant_id, sub.basis)
                    .model_dump(),
                    source="substitute",
                    substitute_position=sub.position,
                    via_forfeit_id=promoting_forfeit.id,
                )
        seat_views.append(
            PromotionSeatView(
                seat_no=seat.seat_no,
                status=seat.status,
                original=original,
                current=current,
                forfeits=[
                    PromotionForfeitView(
                        id=f.id,
                        participant_id=f.forfeited_participant_id,
                        number=participants[f.forfeited_participant_id].number,
                        name=participants[f.forfeited_participant_id].name,
                        created_at=f.created_at,
                    )
                    for f in forfeits_by_seat.get(seat.id, [])
                ],
            )
        )

    return PromotionListResponse(
        id=promotion_list.id,
        tournament_id=promotion_list.tournament_id,
        version=promotion_list.version,
        qualifying_round_no=promotion_list.qualifying_round_no,
        slots=promotion_list.slots,
        expected_event_seq=promotion_list.expected_event_seq,
        created_at=promotion_list.created_at,
        seats=seat_views,
        substitutes=[
            PromotionSubstituteView(
                position=s.position,
                basis=_basis_of(participants, s.participant_id, s.basis),
                status=s.status,
                promoted_seat_no=seat_no_by_id.get(s.promoted_seat_id),
            )
            for s in substitutes
        ],
    )


# ---------- 发布 ----------

def publish_promotion_list(
    db: Session, tournament_id: int, req: PromotionListPublishRequest
) -> tuple[PromotionListResponse, Event | None]:
    """发布一份晋级名单（flush，不提交），返回（响应, 事件）。

    幂等重放时返回（首次响应, None），不产生任何写入；资格不符、
    名额非法或事件序号快照冲突抛 409，且在任何写入之前抛出（不落库）。
    """
    fingerprint = {
        "action": PUBLISH,
        "qualifying_round_no": req.qualifying_round_no,
        "slots": req.slots,
        "expected_event_seq": req.expected_event_seq,
    }
    # 幂等判定先于业务校验：重试已成功请求必须返回原结果
    replay = _find_replay(db, tournament_id, req.request_id, PUBLISH, fingerprint)
    if replay is not None:
        return PromotionListResponse(**replay), None

    # 写锁可能经过等待才取得：以锁内已提交的最新状态重新校验
    rounds = list(
        db.query(Round)
        .filter(Round.tournament_id == tournament_id)
        .order_by(Round.round_no)
    )
    if not rounds:
        raise HTTPException(
            status_code=409, detail="赛事尚未生成任何轮次，不能发布晋级名单"
        )
    open_round = next((r for r in rounds if r.status == "open"), None)
    if open_round is not None:
        raise HTTPException(
            status_code=409,
            detail=f"第 {open_round.round_no} 轮尚未结束，不能发布晋级名单",
        )
    if has_pending_review_for_tournament(db, tournament_id):
        raise HTTPException(
            status_code=409,
            detail="赛事存在待处理的赛果复核，需待复核关闭后才能发布晋级名单",
        )

    last_round = rounds[-1]
    if req.qualifying_round_no != last_round.round_no:
        if any(r.round_no == req.qualifying_round_no for r in rounds):
            raise HTTPException(
                status_code=409,
                detail=(
                    f"资格轮次必须是已生成的最后一轮"
                    f"（第 {last_round.round_no} 轮）"
                ),
            )
        raise HTTPException(
            status_code=409, detail=f"资格轮次 {req.qualifying_round_no} 不存在"
        )

    participant_count = (
        db.query(Participant)
        .filter(Participant.tournament_id == tournament_id)
        .count()
    )
    if req.slots > participant_count:
        raise HTTPException(
            status_code=409,
            detail=f"晋级名额 {req.slots} 超过参赛人数 {participant_count}",
        )

    _check_event_seq_snapshot(db, tournament_id, req.expected_event_seq)

    # 同一事务内重新计算当时的排名积分与 Buchholz（截至资格轮次）：
    # 排名积分降序 -> Buchholz 降序 -> 参赛者编号升序，固定裁决
    records = compute_records(
        db, tournament_id, through_round_no=req.qualifying_round_no
    )

    version = (
        db.query(func.max(PromotionList.version))
        .filter(PromotionList.tournament_id == tournament_id)
        .scalar()
        or 0
    ) + 1
    promotion_list = PromotionList(
        tournament_id=tournament_id,
        version=version,
        qualifying_round_no=req.qualifying_round_no,
        slots=req.slots,
        expected_event_seq=req.expected_event_seq,
    )
    db.add(promotion_list)
    db.flush()  # 取得 list id，供席位/候补/事件使用

    participants = {
        p.id: p
        for p in db.query(Participant)
        .filter(Participant.tournament_id == tournament_id)
        .all()
    }

    seats: list[PromotionSeat] = []
    for rank, record in enumerate(records[: req.slots], start=1):
        seat = PromotionSeat(
            list_id=promotion_list.id,
            seat_no=rank,
            original_participant_id=record.participant_id,
            original_basis=_basis_snapshot(record, rank),
            current_participant_id=record.participant_id,
            status=ACTIVE,
        )
        db.add(seat)
        seats.append(seat)
    substitutes: list[PromotionSubstitute] = []
    for offset, record in enumerate(records[req.slots :], start=1):
        substitute = PromotionSubstitute(
            list_id=promotion_list.id,
            position=offset,
            participant_id=record.participant_id,
            basis=_basis_snapshot(record, req.slots + offset),
            status=WAITING,
        )
        db.add(substitute)
        substitutes.append(substitute)
    db.flush()

    event = append_event(
        db,
        tournament_id,
        PROMOTION_LIST_PUBLISHED,
        {
            "list_id": promotion_list.id,
            "version": version,
            "qualifying_round_no": req.qualifying_round_no,
            "slots": req.slots,
            "expected_event_seq": req.expected_event_seq,
            "seats": [
                {
                    "seat_no": s.seat_no,
                    "participant_id": s.original_participant_id,
                    "number": participants[s.original_participant_id].number,
                }
                for s in seats
            ],
            "substitutes": [
                {
                    "position": s.position,
                    "participant_id": s.participant_id,
                    "number": participants[s.participant_id].number,
                }
                for s in substitutes
            ],
        },
    )

    response = serialize_list(db, promotion_list)
    _save_action(
        db,
        tournament_id,
        promotion_list.id,
        req.request_id,
        PUBLISH,
        fingerprint,
        response.model_dump(mode="json"),
    )
    return response, event


# ---------- 席位放弃 ----------

def forfeit_seat(
    db: Session, list_id: int, seat_no: int, req: PromotionForfeitRequest
) -> tuple[PromotionForfeitResponse, Event | None]:
    """对席位标记放弃并提升最前候补（flush，不提交），返回（响应, 事件）。

    幂等重放时返回（首次响应, None），不产生任何写入；名单/席位/参赛者
    不存在 404，非当前有效晋级者操作或席位已空缺 409，且在任何写入
    之前抛出（不落库）。
    """
    promotion_list = db.get(PromotionList, list_id)
    if promotion_list is None:
        raise HTTPException(status_code=404, detail="晋级名单不存在")
    tournament_id = promotion_list.tournament_id

    fingerprint = {
        "action": FORFEIT,
        "list_id": list_id,
        "seat_no": seat_no,
        "participant_id": req.participant_id,
    }
    # 幂等判定先于业务校验：重试已成功请求必须返回原结果
    replay = _find_replay(db, tournament_id, req.request_id, FORFEIT, fingerprint)
    if replay is not None:
        return PromotionForfeitResponse(**replay), None

    seat = (
        db.query(PromotionSeat)
        .filter(
            PromotionSeat.list_id == list_id,
            PromotionSeat.seat_no == seat_no,
        )
        .first()
    )
    if seat is None:
        raise HTTPException(status_code=404, detail="席位不存在")
    participant = (
        db.query(Participant)
        .filter(
            Participant.id == req.participant_id,
            Participant.tournament_id == tournament_id,
        )
        .first()
    )
    if participant is None:
        raise HTTPException(status_code=404, detail="参赛者不存在")
    if seat.status != ACTIVE or seat.current_participant_id is None:
        raise HTTPException(
            status_code=409, detail="该席位已空缺，没有可放弃的晋级者"
        )
    if seat.current_participant_id != req.participant_id:
        raise HTTPException(
            status_code=409, detail="只有该席位当前有效晋级者可以标记放弃"
        )

    # 最前候补：position 最小的 waiting 候补
    substitute = (
        db.query(PromotionSubstitute)
        .filter(
            PromotionSubstitute.list_id == list_id,
            PromotionSubstitute.status == WAITING,
        )
        .order_by(PromotionSubstitute.position)
        .first()
    )

    # 放弃者若是此前提升的候补，其候补行标记为「曾提升，后又放弃」
    if seat.current_participant_id != seat.original_participant_id:
        current_substitute = (
            db.query(PromotionSubstitute)
            .filter(
                PromotionSubstitute.list_id == list_id,
                PromotionSubstitute.participant_id == seat.current_participant_id,
            )
            .first()
        )
        if current_substitute is not None:
            current_substitute.status = FORFEITED

    forfeit = PromotionForfeit(
        list_id=list_id,
        seat_id=seat.id,
        forfeited_participant_id=participant.id,
        promoted_substitute_id=substitute.id if substitute is not None else None,
    )
    db.add(forfeit)

    promoted_participant: Participant | None = None
    if substitute is not None:
        # 同一事务内把最前候补提升为该席位
        substitute.status = PROMOTED
        substitute.promoted_seat_id = seat.id
        seat.current_participant_id = substitute.participant_id
        promoted_participant = db.get(Participant, substitute.participant_id)
    else:
        # 空缺无候补：保持空席
        seat.current_participant_id = None
        seat.status = VACANT
    db.flush()  # 取得 forfeit.id 与 created_at，供事件与响应使用

    event = append_event(
        db,
        tournament_id,
        PROMOTION_LIST_SUBSTITUTE_PROMOTED,
        {
            "list_id": promotion_list.id,
            "version": promotion_list.version,
            "seat_no": seat.seat_no,
            "forfeit_id": forfeit.id,
            "forfeited_participant_id": participant.id,
            "forfeited_number": participant.number,
            "promoted_participant_id": (
                promoted_participant.id if promoted_participant else None
            ),
            "promoted_number": (
                promoted_participant.number if promoted_participant else None
            ),
            "substitute_position": (
                substitute.position if substitute is not None else None
            ),
            "seat_status": seat.status,
        },
    )

    response = PromotionForfeitResponse(
        id=forfeit.id,
        list_id=promotion_list.id,
        tournament_id=tournament_id,
        version=promotion_list.version,
        seat_no=seat.seat_no,
        forfeited=PromotionParticipantRef(
            participant_id=participant.id,
            number=participant.number,
            name=participant.name,
        ),
        promoted=(
            PromotionParticipantRef(
                participant_id=promoted_participant.id,
                number=promoted_participant.number,
                name=promoted_participant.name,
            )
            if promoted_participant
            else None
        ),
        promoted_substitute_position=(
            substitute.position if substitute is not None else None
        ),
        seat_status=seat.status,
        created_at=forfeit.created_at,
    )
    _save_action(
        db,
        tournament_id,
        promotion_list.id,
        req.request_id,
        FORFEIT,
        fingerprint,
        response.model_dump(mode="json"),
    )
    return response, event


# ---------- 查询 ----------

def list_promotion_lists(
    db: Session, tournament_id: int
) -> list[PromotionList]:
    """赛事下的全部晋级名单（按版本升序）。"""
    return list(
        db.query(PromotionList)
        .filter(PromotionList.tournament_id == tournament_id)
        .order_by(PromotionList.version)
    )


def get_promotion_list_by_version(
    db: Session, tournament_id: int, version: int
) -> PromotionList:
    lst = (
        db.query(PromotionList)
        .filter(
            PromotionList.tournament_id == tournament_id,
            PromotionList.version == version,
        )
        .first()
    )
    if lst is None:
        raise HTTPException(status_code=404, detail="晋级名单版本不存在")
    return lst
