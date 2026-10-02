"""晋级名单模块测试。

覆盖需求：
- 发布：仅允许在「所有已生成轮次结束且没有待处理赛果复核」时发布；
  request_id 幂等键 + expected_event_seq 乐观快照 + 资格轮次（须为
  已生成的最后一轮）+ 晋级名额；同一事务内重算排名积分与 Buchholz，
  按排名积分、Buchholz、参赛者编号固定裁决，冻结前 N 名席位与后续
  候补顺序；事件序号变化、名额非法、轮次未完成一律 409 且不留记录；
- 冻结：发布后原排名与扣分变化不改写该版本；再次发布生成新版本；
- 放弃：仅当前有效晋级者可操作，同一事务内把最前候补提升为该席位；
  重复 request_id 返回首次结果；空缺无候补时保持空席；
- 查询：赛事/版本查询包含每席位原始与当前参赛者、状态和依据；
- 事件：promotion_list.published / promotion_list.substitute_promoted
  进入事件流，被拒绝的请求不产生事件；
- 并发：并发放弃同一席位仅一成一败；并发同键不同参数发布仅一成。
"""
import concurrent.futures
import threading

import pytest


# ---------- 基础工具 ----------

def create_tournament(client, name="晋级测试赛"):
    r = client.post("/tournaments", json={"name": name})
    assert r.status_code == 201
    return r.json()["id"]


def add_players(client, tid, names):
    r = client.post(
        f"/tournaments/{tid}/participants/bulk", json={"names": names}
    )
    assert r.status_code == 201
    return r.json()


def next_round(client, tid):
    r = client.post(f"/tournaments/{tid}/rounds/next")
    assert r.status_code == 201, r.text
    return r.json()


def put_result(client, match, sw, sb):
    mid = match["id"] if isinstance(match, dict) else match
    r = client.put(
        f"/matches/{mid}/result",
        json={"score_white": sw, "score_black": sb},
    )
    assert r.status_code == 200, r.text
    return r.json()


def event_seq(client, tid):
    return client.get(f"/tournaments/{tid}/events").json()["last_seq"]


def events_of(client, tid, type_=None):
    items = client.get(f"/tournaments/{tid}/events?limit=200").json()["items"]
    return [e for e in items if type_ is None or e["type"] == type_]


def standings(client, tid):
    return client.get(f"/tournaments/{tid}/standings").json()


def publish(client, tid, **body):
    return client.post(f"/tournaments/{tid}/promotion-lists", json=body)


def forfeit(client, list_id, seat_no, **body):
    return client.post(
        f"/promotion-lists/{list_id}/seats/{seat_no}/forfeit", json=body
    )


def get_list(client, tid, version):
    return client.get(f"/tournaments/{tid}/promotion-lists/{version}")


def list_lists(client, tid):
    return client.get(f"/tournaments/{tid}/promotion-lists")


def setup_closed_round(client, n=5):
    """n 人赛事，第 1 轮全部以白胜录入并封盘。返回 (tid, players, real)。"""
    tid = create_tournament(client)
    players = add_players(client, tid, [f"选手{i}" for i in range(1, n + 1)])
    rnd = next_round(client, tid)
    real = [m for m in rnd["matches"] if not m["is_bye"]]
    for m in real:
        put_result(client, m, 1.0, 0.0)
    rounds = client.get(f"/tournaments/{tid}/rounds").json()
    assert rounds[-1]["status"] == "closed"
    return tid, players, real


def publish_body(client, tid, **over):
    body = {
        "request_id": "promo-1",
        "expected_event_seq": event_seq(client, tid),
        "qualifying_round_no": 1,
        "slots": 2,
    }
    body.update(over)
    return body


def published(client, tid, **over):
    """成功发布一份名单，返回响应 JSON。"""
    r = publish(client, tid, **publish_body(client, tid, **over))
    assert r.status_code == 201, r.text
    return r.json()


# ---------- 一、发布 ----------

def test_publish_success_freezes_seats_substitutes_and_basis(client):
    tid, players, real = setup_closed_round(client, n=5)
    seq = event_seq(client, tid)
    expected = standings(client, tid)  # 发布时的排名口径

    r = publish(client, tid, **publish_body(client, tid, slots=2))
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["tournament_id"] == tid
    assert body["version"] == 1
    assert body["qualifying_round_no"] == 1
    assert body["slots"] == 2
    assert body["expected_event_seq"] == seq
    assert body["created_at"] is not None

    # 前 2 名为席位，其余 3 人按排名冻结为候补；依据与当时排名一致
    assert [s["seat_no"] for s in body["seats"]] == [1, 2]
    for i, seat in enumerate(body["seats"]):
        row = expected[i]
        assert seat["status"] == "active"
        assert seat["original"]["participant_id"] == row["participant_id"]
        assert seat["original"]["rank"] == row["rank"]
        assert seat["original"]["points"] == row["points"]
        assert seat["original"]["penalty"] == row["penalty"]
        assert seat["original"]["rank_points"] == row["rank_points"]
        assert seat["original"]["buchholz"] == row["buchholz"]
        # 当前持有者即原始晋级者
        assert seat["current"]["participant_id"] == row["participant_id"]
        assert seat["current"]["source"] == "original"
        assert seat["current"]["substitute_position"] is None
        assert seat["current"]["via_forfeit_id"] is None
        assert seat["forfeits"] == []

    assert [s["position"] for s in body["substitutes"]] == [1, 2, 3]
    for j, sub in enumerate(body["substitutes"]):
        row = expected[2 + j]
        assert sub["basis"]["participant_id"] == row["participant_id"]
        assert sub["basis"]["rank"] == row["rank"]
        assert sub["basis"]["rank_points"] == row["rank_points"]
        assert sub["basis"]["buchholz"] == row["buchholz"]
        assert sub["status"] == "waiting"
        assert sub["promoted_seat_no"] is None

    # 事件同事务写入，序号恰好 +1
    assert event_seq(client, tid) == seq + 1
    evs = events_of(client, tid, "promotion_list.published")
    assert len(evs) == 1
    payload = evs[0]["payload"]
    assert payload["list_id"] == body["id"]
    assert payload["version"] == 1
    assert payload["qualifying_round_no"] == 1
    assert payload["slots"] == 2
    assert [s["seat_no"] for s in payload["seats"]] == [1, 2]
    assert [s["position"] for s in payload["substitutes"]] == [1, 2, 3]


def test_publish_fixed_tiebreak_by_number(client):
    # 首轮全员同分（未赛）时无法发布（轮次未结束）；
    # 构造同分同 Buchholz：2 人赛事一轮和棋，排名积分与 Buchholz 均相同，
    # 编号小者居首
    tid = create_tournament(client)
    add_players(client, tid, ["甲", "乙"])
    rnd = next_round(client, tid)
    put_result(client, rnd["matches"][0], 0.5, 0.5)

    body = published(client, tid, slots=1)
    assert body["seats"][0]["original"]["number"] == 1
    assert body["substitutes"][0]["basis"]["number"] == 2


def test_publish_rejected_when_round_open(client):
    tid = create_tournament(client)
    add_players(client, tid, ["甲", "乙", "丙", "丁"])
    next_round(client, tid)  # 轮次 open，未录赛果
    seq = event_seq(client, tid)

    r = publish(client, tid, **publish_body(client, tid))
    assert r.status_code == 409
    assert "尚未结束" in r.json()["detail"]
    assert event_seq(client, tid) == seq  # 不落库、不占号
    assert list_lists(client, tid).json() == []


def test_publish_rejected_with_pending_review(client):
    tid, players, real = setup_closed_round(client)
    r = client.post(
        f"/matches/{real[0]['id']}/result-reviews",
        json={
            "request_id": "rev-1",
            "expected_event_seq": event_seq(client, tid),
            "score_white": 0.0,
            "score_black": 1.0,
            "reason": "记分表登记有误",
        },
    )
    assert r.status_code == 201
    seq = event_seq(client, tid)

    r = publish(client, tid, **publish_body(client, tid))
    assert r.status_code == 409
    assert "复核" in r.json()["detail"]
    assert event_seq(client, tid) == seq
    assert list_lists(client, tid).json() == []


def test_publish_rejected_without_any_round(client):
    tid = create_tournament(client)
    add_players(client, tid, ["甲", "乙"])
    seq = event_seq(client, tid)

    r = publish(client, tid, **publish_body(client, tid))
    assert r.status_code == 409
    assert "尚未生成任何轮次" in r.json()["detail"]
    assert event_seq(client, tid) == seq
    assert list_lists(client, tid).json() == []


def test_publish_qualifying_round_must_be_last_generated(client):
    tid, players, real = setup_closed_round(client)
    next_round(client, tid)  # 第 2 轮 open
    for m in [m for m in client.get(f"/tournaments/{tid}/rounds").json()[-1]["matches"] if not m["is_bye"]]:
        put_result(client, m, 1.0, 0.0)
    seq = event_seq(client, tid)

    # 资格轮次不是最后一轮
    r = publish(client, tid, **publish_body(client, tid, qualifying_round_no=1))
    assert r.status_code == 409
    assert "最后一轮" in r.json()["detail"]
    # 资格轮次不存在
    r = publish(client, tid, **publish_body(client, tid, qualifying_round_no=9))
    assert r.status_code == 409
    assert "不存在" in r.json()["detail"]
    # 最后一轮可以发布
    r = publish(
        client, tid, **publish_body(client, tid, qualifying_round_no=2)
    )
    assert r.status_code == 201, r.text
    assert r.json()["qualifying_round_no"] == 2
    assert event_seq(client, tid) == seq + 1


def test_publish_slots_validation(client):
    tid, players, real = setup_closed_round(client, n=4)
    seq = event_seq(client, tid)

    # 名额为 0 / 负数：schema 422
    r = publish(client, tid, **publish_body(client, tid, slots=0))
    assert r.status_code == 422
    # 名额超过参赛人数：409 且不落库
    r = publish(client, tid, **publish_body(client, tid, slots=5))
    assert r.status_code == 409
    assert "超过参赛人数" in r.json()["detail"]

    assert event_seq(client, tid) == seq
    assert list_lists(client, tid).json() == []


def test_publish_snapshot_conflict_rejects_without_write(client):
    tid, players, real = setup_closed_round(client)
    seq = event_seq(client, tid)

    r = publish(
        client, tid, **publish_body(client, tid, expected_event_seq=seq - 1)
    )
    assert r.status_code == 409
    assert "快照" in r.json()["detail"]
    assert event_seq(client, tid) == seq
    assert list_lists(client, tid).json() == []


def test_publish_idempotent_replay(client):
    tid, players, real = setup_closed_round(client)
    body = publish_body(client, tid)
    r1 = publish(client, tid, **body)
    assert r1.status_code == 201
    seq = event_seq(client, tid)

    r2 = publish(client, tid, **body)
    assert r2.status_code == 201
    assert r2.json() == r1.json()
    assert event_seq(client, tid) == seq  # 无新事件、不占号
    assert len(events_of(client, tid, "promotion_list.published")) == 1
    assert len(list_lists(client, tid).json()) == 1


def test_publish_same_request_id_different_params_conflict(client):
    tid, players, real = setup_closed_round(client)
    body = publish_body(client, tid)
    publish(client, tid, **body)
    seq = event_seq(client, tid)

    r = publish(client, tid, **{**body, "slots": 3})
    assert r.status_code == 409
    r = publish(client, tid, **{**body, "expected_event_seq": seq})
    assert r.status_code == 409
    assert event_seq(client, tid) == seq
    assert len(list_lists(client, tid).json()) == 1


def test_publish_frozen_against_later_ranking_and_penalty_changes(client):
    tid, players, real = setup_closed_round(client, n=4)
    body = published(client, tid, slots=2)
    original_seats = body["seats"]

    # 扣分改变当前排名：给头名扣 2 分使其垫底
    client.post(
        f"/tournaments/{tid}/penalties",
        json={
            "request_id": "pen-1",
            "expected_event_seq": event_seq(client, tid),
            "participant_id": original_seats[0]["original"]["participant_id"],
            "points": 2.0,
            "reason": "严重违规",
        },
    )
    changed = standings(client, tid)
    assert changed[0]["participant_id"] != original_seats[0]["original"]["participant_id"]

    # 已发布版本不改写：席位与依据保持发布时快照
    frozen = get_list(client, tid, 1).json()
    assert frozen["seats"] == original_seats
    assert frozen["seats"][0]["original"]["penalty"] == 0.0
    assert frozen["substitutes"] == body["substitutes"]


def test_publish_new_version_reflects_later_state(client):
    tid, players, real = setup_closed_round(client, n=4)
    v1 = published(client, tid, slots=2)

    # 扣分后再次发布：生成新版本，依据按当时重算
    client.post(
        f"/tournaments/{tid}/penalties",
        json={
            "request_id": "pen-1",
            "expected_event_seq": event_seq(client, tid),
            "participant_id": v1["seats"][0]["original"]["participant_id"],
            "points": 1.0,
            "reason": "迟到",
        },
    )
    v2 = published(
        client, tid, request_id="promo-2", slots=2,
    )
    assert v2["version"] == 2
    assert v2["seats"] != v1["seats"]

    # 旧版本原样保留
    assert get_list(client, tid, 1).json()["seats"] == v1["seats"]
    assert [lst["version"] for lst in list_lists(client, tid).json()] == [1, 2]


def test_publish_tournament_not_found(client):
    body = {
        "request_id": "promo-1",
        "expected_event_seq": 0,
        "qualifying_round_no": 1,
        "slots": 1,
    }
    r = publish(client, 9999, **body)
    assert r.status_code == 404


# ---------- 二、席位放弃与候补提升 ----------

def test_forfeit_promotes_earliest_substitute(client):
    tid, players, real = setup_closed_round(client, n=5)
    body = published(client, tid, slots=2)
    list_id = body["id"]
    first_sub = body["substitutes"][0]["basis"]
    holder = body["seats"][0]["current"]["participant_id"]
    seq = event_seq(client, tid)

    r = forfeit(
        client, list_id, 1,
        request_id="forfeit-1", participant_id=holder,
    )
    assert r.status_code == 201, r.text
    fv = r.json()
    assert fv["list_id"] == list_id
    assert fv["version"] == 1
    assert fv["seat_no"] == 1
    assert fv["forfeited"]["participant_id"] == holder
    assert fv["promoted"]["participant_id"] == first_sub["participant_id"]
    assert fv["promoted_substitute_position"] == 1
    assert fv["seat_status"] == "active"
    assert fv["created_at"] is not None

    # 事件同事务写入，序号恰好 +1
    assert event_seq(client, tid) == seq + 1
    evs = events_of(client, tid, "promotion_list.substitute_promoted")
    assert len(evs) == 1
    payload = evs[0]["payload"]
    assert payload["list_id"] == list_id
    assert payload["seat_no"] == 1
    assert payload["forfeited_participant_id"] == holder
    assert payload["promoted_participant_id"] == first_sub["participant_id"]
    assert payload["substitute_position"] == 1
    assert payload["seat_status"] == "active"

    # 版本详情：席位当前持有者变为最前候补，依据为候补的冻结排名
    detail = get_list(client, tid, 1).json()
    seat1 = detail["seats"][0]
    assert seat1["status"] == "active"
    assert seat1["original"]["participant_id"] == holder  # 原始晋级者不改写
    assert seat1["current"]["participant_id"] == first_sub["participant_id"]
    assert seat1["current"]["source"] == "substitute"
    assert seat1["current"]["substitute_position"] == 1
    assert seat1["current"]["via_forfeit_id"] == fv["id"]
    assert seat1["current"]["rank_points"] == first_sub["rank_points"]
    assert [f["participant_id"] for f in seat1["forfeits"]] == [holder]
    # 候补状态
    sub1 = detail["substitutes"][0]
    assert sub1["status"] == "promoted"
    assert sub1["promoted_seat_no"] == 1
    assert detail["substitutes"][1]["status"] == "waiting"


def test_forfeit_requires_current_holder(client):
    tid, players, real = setup_closed_round(client, n=4)
    body = published(client, tid, slots=2)
    list_id = body["id"]
    holder = body["seats"][0]["current"]["participant_id"]
    other = body["seats"][1]["current"]["participant_id"]
    seq = event_seq(client, tid)

    # 非当前有效晋级者（另一席位的持有者）不能放弃本席位
    r = forfeit(
        client, list_id, 1,
        request_id="forfeit-x", participant_id=other,
    )
    assert r.status_code == 409
    assert "当前有效晋级者" in r.json()["detail"]

    # 候补也不是当前有效晋级者
    sub = body["substitutes"][0]["basis"]["participant_id"]
    r = forfeit(
        client, list_id, 1,
        request_id="forfeit-y", participant_id=sub,
    )
    assert r.status_code == 409

    # 参赛者不存在
    r = forfeit(
        client, list_id, 1,
        request_id="forfeit-z", participant_id=9999,
    )
    assert r.status_code == 404

    assert event_seq(client, tid) == seq  # 均不落库、不占号
    detail = get_list(client, tid, 1).json()
    assert detail["seats"][0]["current"]["participant_id"] == holder


def test_forfeit_idempotent_replay(client):
    tid, players, real = setup_closed_round(client, n=4)
    body = published(client, tid, slots=2)
    list_id = body["id"]
    holder = body["seats"][0]["current"]["participant_id"]

    req = {"request_id": "forfeit-1", "participant_id": holder}
    r1 = forfeit(client, list_id, 1, **req)
    assert r1.status_code == 201
    seq = event_seq(client, tid)

    r2 = forfeit(client, list_id, 1, **req)
    assert r2.status_code == 201
    assert r2.json() == r1.json()
    assert event_seq(client, tid) == seq
    evs = events_of(client, tid, "promotion_list.substitute_promoted")
    assert len(evs) == 1
    detail = get_list(client, tid, 1).json()
    assert len(detail["seats"][0]["forfeits"]) == 1

    # 同键不同参数：409
    r = forfeit(
        client, list_id, 1,
        request_id="forfeit-1",
        participant_id=body["seats"][1]["current"]["participant_id"],
    )
    assert r.status_code == 409
    assert event_seq(client, tid) == seq


def test_forfeit_chain_promotes_next_substitute(client):
    tid, players, real = setup_closed_round(client, n=5)
    body = published(client, tid, slots=1)
    list_id = body["id"]
    holder = body["seats"][0]["current"]["participant_id"]

    # 原始晋级者放弃 -> 候补 1 提升
    r = forfeit(client, list_id, 1, request_id="f-1", participant_id=holder)
    assert r.status_code == 201
    promoted1 = r.json()["promoted"]["participant_id"]
    assert promoted1 == body["substitutes"][0]["basis"]["participant_id"]

    # 提升后的候补是当前有效晋级者，也可以放弃 -> 候补 2 提升
    r = forfeit(client, list_id, 1, request_id="f-2", participant_id=promoted1)
    assert r.status_code == 201, r.text
    assert r.json()["promoted"]["participant_id"] == (
        body["substitutes"][1]["basis"]["participant_id"]
    )

    detail = get_list(client, tid, 1).json()
    seat = detail["seats"][0]
    assert seat["current"]["participant_id"] == (
        body["substitutes"][1]["basis"]["participant_id"]
    )
    assert seat["current"]["substitute_position"] == 2
    assert [f["participant_id"] for f in seat["forfeits"]] == [holder, promoted1]
    subs = {s["position"]: s for s in detail["substitutes"]}
    assert subs[1]["status"] == "forfeited"  # 曾提升，后又放弃
    assert subs[1]["promoted_seat_no"] == 1
    assert subs[2]["status"] == "promoted"
    assert subs[3]["status"] == "waiting"


def test_forfeit_without_substitute_keeps_vacant_seat(client):
    tid, players, real = setup_closed_round(client, n=3)
    # 名额等于参赛人数：没有候补
    body = published(client, tid, slots=3)
    assert body["substitutes"] == []
    list_id = body["id"]
    holder = body["seats"][0]["current"]["participant_id"]
    seq = event_seq(client, tid)

    r = forfeit(client, list_id, 1, request_id="f-1", participant_id=holder)
    assert r.status_code == 201, r.text
    fv = r.json()
    assert fv["promoted"] is None
    assert fv["promoted_substitute_position"] is None
    assert fv["seat_status"] == "vacant"

    evs = events_of(client, tid, "promotion_list.substitute_promoted")
    assert len(evs) == 1
    assert evs[0]["payload"]["promoted_participant_id"] is None
    assert evs[0]["payload"]["seat_status"] == "vacant"

    detail = get_list(client, tid, 1).json()
    seat = detail["seats"][0]
    assert seat["status"] == "vacant"
    assert seat["current"] is None
    assert seat["original"]["participant_id"] == holder

    # 空席没有当前有效晋级者，不能再放弃
    r = forfeit(client, list_id, 1, request_id="f-2", participant_id=holder)
    assert r.status_code == 409
    assert "空缺" in r.json()["detail"]
    assert event_seq(client, tid) == seq + 1


def test_forfeit_exhausts_substitutes_then_vacant(client):
    tid, players, real = setup_closed_round(client, n=3)
    body = published(client, tid, slots=1)
    list_id = body["id"]
    holder = body["seats"][0]["current"]["participant_id"]

    # 原始持有者放弃 -> 候补 1 提升；候补 1 再放弃 -> 候补 2 提升；
    # 候补 2 再放弃 -> 无候补，空席
    r = forfeit(client, list_id, 1, request_id="f-1", participant_id=holder)
    p1 = r.json()["promoted"]["participant_id"]
    r = forfeit(client, list_id, 1, request_id="f-2", participant_id=p1)
    p2 = r.json()["promoted"]["participant_id"]
    r = forfeit(client, list_id, 1, request_id="f-3", participant_id=p2)
    assert r.json()["seat_status"] == "vacant"
    assert r.json()["promoted"] is None

    detail = get_list(client, tid, 1).json()
    assert detail["seats"][0]["status"] == "vacant"
    assert detail["seats"][0]["current"] is None
    assert len(detail["seats"][0]["forfeits"]) == 3


def test_forfeit_unknown_list_or_seat(client):
    tid, players, real = setup_closed_round(client, n=3)
    body = published(client, tid, slots=1)
    holder = body["seats"][0]["current"]["participant_id"]

    r = forfeit(client, 9999, 1, request_id="f-1", participant_id=holder)
    assert r.status_code == 404
    r = forfeit(client, body["id"], 9, request_id="f-1", participant_id=holder)
    assert r.status_code == 404
    assert r.json()["detail"] == "席位不存在"


def test_forfeit_does_not_affect_other_versions(client):
    tid, players, real = setup_closed_round(client, n=4)
    v1 = published(client, tid, slots=2)
    v2 = published(client, tid, request_id="promo-2", slots=2)

    holder = v2["seats"][0]["current"]["participant_id"]
    r = forfeit(client, v2["id"], 1, request_id="f-1", participant_id=holder)
    assert r.status_code == 201

    # v1 不受影响
    assert get_list(client, tid, 1).json()["seats"] == v1["seats"]
    # v2 已更新
    seat = get_list(client, tid, 2).json()["seats"][0]
    assert seat["current"]["source"] == "substitute"


# ---------- 三、查询 ----------

def test_queries_by_tournament_and_version(client):
    tid, players, real = setup_closed_round(client, n=4)
    assert list_lists(client, tid).json() == []

    published(client, tid, slots=2)
    published(client, tid, request_id="promo-2", slots=3)

    r = list_lists(client, tid)
    assert r.status_code == 200
    body = r.json()
    assert [lst["version"] for lst in body] == [1, 2]
    assert body[0]["slots"] == 2
    assert body[1]["slots"] == 3
    assert len(body[1]["seats"]) == 3
    assert len(body[1]["substitutes"]) == 1

    r = get_list(client, tid, 2)
    assert r.status_code == 200
    assert r.json()["version"] == 2

    # 版本不存在 / 赛事不存在
    assert get_list(client, tid, 9).status_code == 404
    assert get_list(client, 9999, 1).status_code == 404
    assert list_lists(client, 9999).status_code == 404


# ---------- 四、并发 ----------

def _run_concurrently(fn, n, barrier):
    results: list = [None] * n
    errors: list[BaseException] = []

    def worker(i):
        try:
            barrier.wait(timeout=10)
            results[i] = fn(i)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(worker, i) for i in range(n)]
        concurrent.futures.wait(futs, timeout=30)
    assert not errors, errors
    return results


def test_concurrent_forfeit_only_one_succeeds(live_server_factory):
    """并发放弃同一席位（同一持有者、不同 request_id）：恰好一次成功。"""
    with live_server_factory() as srv:
        client = srv.client()
        tid, players, real = setup_closed_round(client, n=4)
        body = published(client, tid, slots=2)
        list_id = body["id"]
        holder = body["seats"][0]["current"]["participant_id"]
        seq = event_seq(client, tid)
        n = 6
        barrier = threading.Barrier(n)

        def call(i):
            local = srv.client()
            try:
                barrier.wait(timeout=10)
                r = local.post(
                    f"/promotion-lists/{list_id}/seats/1/forfeit",
                    json={"request_id": f"f-{i}", "participant_id": holder},
                )
                return r.status_code
            finally:
                local.close()

        statuses = _run_concurrently(call, n, barrier)
        assert statuses.count(201) == 1
        assert statuses.count(409) == n - 1

        detail = get_list(client, tid, 1).json()
        assert len(detail["seats"][0]["forfeits"]) == 1
        assert event_seq(client, tid) == seq + 1  # 仅一次放弃占一个序号


def test_concurrent_publish_same_key_conflict(live_server_factory):
    """并发以同一 request_id 提交不同参数：恰好一份名单落库，其余 409。"""
    with live_server_factory() as srv:
        client = srv.client()
        tid, players, real = setup_closed_round(client, n=4)
        seq = event_seq(client, tid)
        n = 6
        barrier = threading.Barrier(n)

        def call(i):
            local = srv.client()
            try:
                barrier.wait(timeout=10)
                r = local.post(
                    f"/tournaments/{tid}/promotion-lists",
                    json=publish_body(
                        client, tid, request_id="same-key", slots=i + 1
                    ),
                )
                return r.status_code
            finally:
                local.close()

        statuses = _run_concurrently(call, n, barrier)
        assert statuses.count(201) == 1
        assert statuses.count(409) == n - 1
        assert len(list_lists(client, tid).json()) == 1
        assert event_seq(client, tid) == seq + 1


# ---------- 实时服务器夹具（与 test_penalties 同款，后台 uvicorn） ----------

@pytest.fixture()
def live_server_factory():
    import socket
    import time

    import httpx
    import uvicorn

    from app.database import Base, engine
    from app.main import app

    def wait_until(cond, timeout=10.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if cond():
                return
            time.sleep(0.01)
        raise AssertionError("server not started")

    class Server:
        def __init__(self):
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            self.port = sock.getsockname()[1]
            sock.close()
            self.server = uvicorn.Server(
                uvicorn.Config(
                    app,
                    host="127.0.0.1",
                    port=self.port,
                    log_level="error",
                    access_log=False,
                )
            )
            self.thread = threading.Thread(target=self.server.run, daemon=True)

        def client(self):
            return httpx.Client(
                base_url=f"http://127.0.0.1:{self.port}", timeout=10.0
            )

        def __enter__(self):
            Base.metadata.drop_all(bind=engine)
            Base.metadata.create_all(bind=engine)
            self.thread.start()
            wait_until(lambda: self.server.started)
            return self

        def __exit__(self, *exc):
            self.server.should_exit = True
            self.thread.join(timeout=10)

    return Server
