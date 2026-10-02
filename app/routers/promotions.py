"""晋级名单接口。

- POST /tournaments/{id}/promotion-lists：发布一份晋级名单。请求携带
  幂等键 request_id、乐观快照 expected_event_seq、资格轮次（须为已
  生成的最后一轮）与晋级名额 N；仅允许在所有已生成轮次结束且没有
  待处理赛果复核时发布。成功时在同一写事务内重算排名积分与
  Buchholz、固定裁决并冻结前 N 名席位与候补顺序，写入
  promotion_list.published 事件。
- POST /promotion-lists/{id}/seats/{seat_no}/forfeit：对已冻结名单的
  席位标记放弃（仅该席位当前有效晋级者可操作），同一事务内把最前
  候补提升为该席位并写入 promotion_list.substitute_promoted 事件；
  无候补时保持空席。
- GET /tournaments/{id}/promotion-lists：赛事下全部版本（含每席位
  原始与当前参赛者、状态与冻结依据、候补顺序）。
- GET /tournaments/{id}/promotion-lists/{version}：指定版本详情。

重复 request_id（参数一致）原样返回首次结果，不产生新写入与事件；
同键不同参数返回 409。轮次未结束、有待处理复核、资格轮次不符、
名额非法、快照冲突一律 409 且不落库。
"""
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from ..database import get_db
from ..events.service import commit_and_publish
from ..locking import writer_lock
from ..models import PromotionList
from ..schemas import (
    PromotionForfeitRequest,
    PromotionForfeitResponse,
    PromotionListPublishRequest,
    PromotionListResponse,
)
from ..services.promotions import (
    forfeit_seat,
    get_promotion_list_by_version,
    list_promotion_lists,
    publish_promotion_list,
    serialize_list,
)
from .tournaments import get_tournament_or_404

router = APIRouter()


@router.post(
    "/tournaments/{tournament_id}/promotion-lists",
    response_model=PromotionListResponse,
    status_code=201,
)
def publish(
    tournament_id: int,
    payload: PromotionListPublishRequest,
    db: Session = Depends(get_db),
):
    get_tournament_or_404(db, tournament_id)
    # 写请求进程级串行：资格校验、快照确认、排名重算与冻结、事件序号
    # 与「提交→发布」全程持锁
    with writer_lock():
        db.expire_all()
        response, event = publish_promotion_list(db, tournament_id, payload)
        # 幂等重放（event 为 None）没有任何写入，直接返回首次结果
        if event is not None:
            commit_and_publish(db, [event])
    return response


@router.post(
    "/promotion-lists/{list_id}/seats/{seat_no}/forfeit",
    response_model=PromotionForfeitResponse,
    status_code=201,
)
def forfeit(
    list_id: int,
    seat_no: int,
    payload: PromotionForfeitRequest,
    db: Session = Depends(get_db),
):
    # 存在性先在锁外给出 404；锁内会以最新状态重新读取并校验
    if db.get(PromotionList, list_id) is None:
        raise HTTPException(status_code=404, detail="晋级名单不存在")

    with writer_lock():
        db.expire_all()
        response, event = forfeit_seat(db, list_id, seat_no, payload)
        # 幂等重放（event 为 None）没有任何写入，直接返回首次结果
        if event is not None:
            commit_and_publish(db, [event])
    return response


@router.get(
    "/tournaments/{tournament_id}/promotion-lists",
    response_model=list[PromotionListResponse],
)
def list_lists(tournament_id: int, db: Session = Depends(get_db)):
    get_tournament_or_404(db, tournament_id)
    return [
        serialize_list(db, lst)
        for lst in list_promotion_lists(db, tournament_id)
    ]


@router.get(
    "/tournaments/{tournament_id}/promotion-lists/{version}",
    response_model=PromotionListResponse,
)
def get_list(tournament_id: int, version: int, db: Session = Depends(get_db)):
    get_tournament_or_404(db, tournament_id)
    return serialize_list(
        db, get_promotion_list_by_version(db, tournament_id, version)
    )
