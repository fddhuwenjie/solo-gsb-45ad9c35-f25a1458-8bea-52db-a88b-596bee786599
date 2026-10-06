"""家庭应收台账行为测试（多会话凭证按账户汇成净应收）。

覆盖：
- 计费栏入账方向：full/+全额、supplement/+应补、refund/−应退、no_adjustment/0；
  余额允许为负；
- 一个会话从首张凭证起按 voucher_no 连续记入同一账户，同批可按序补齐前序；
- 不存在的会话 / 未出单序号 → ledger_voucher_not_found，整批拒绝；
- 跳号（不从 1 起 / 中间缺号）→ ledger_voucher_gap，整批拒绝；
- 跨账户（凭证已记别的账户）→ ledger_voucher_cross_account，整批拒绝；
- 重复（历史已记 / 批内重复 / 并发抢同一凭证）→ 整批拒绝，不重复计费；
- 整批原子性：批内任一凭证不合法，已在同批处理的前序不落盘、余额不变；
- 并发入账不同凭证串行成功、序号与余额一致；同凭证并发只有一批成功；
- 更正/重试出单不改旧入账，新凭证可接着入账；
- 按入账次序查询明细与余额；空账户为 0；重启后明细与余额仍在；
- 旧库（无台账表）打开自动补建；既有出单/更正/会话接口行为不变。
"""

import threading

import pytest
from fastapi.testclient import TestClient

from app.ledger import post_vouchers
from app.main import app
from app.storage import configure_store

client = TestClient(app)


DISPATCH = dict(
    prices=[0.0, 1.0, 1.0, 0.0],
    base_load_kw=[0.2, 0.2, 0.2, 0.2],
    pv_kw=[0.0, 0.0, 0.0, 0.0],
    capacity_kwh=0.4,
    max_charge_kw=1.0,
    max_discharge_kw=1.0,
    charge_efficiency=0.9,
    discharge_efficiency=0.9,
    initial_soc_kwh=0.0,
    min_terminal_soc_kwh=0.0,
    degradation_cost_per_kwh=0.2,
)

RECORDS = [
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=round(0.2 / 0.9, 6), discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=round(0.1 + 0.2 / 0.9, 6),
        soc_start_kwh=0.0, soc_end_kwh=0.2,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.09,
        grid_purchase_energy_kwh=0.01,
        soc_start_kwh=0.2, soc_end_kwh=0.1,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.09,
        grid_purchase_energy_kwh=0.01,
        soc_start_kwh=0.1, soc_end_kwh=0.0,
    ),
    dict(
        base_load_kw=0.2, pv_kw=0.0,
        charge_ac_energy_kwh=0.0, discharge_ac_energy_kwh=0.0,
        grid_purchase_energy_kwh=0.1,
        soc_start_kwh=0.0, soc_end_kwh=0.0,
    ),
]

TOL = 0.01
ORIG_TOTAL = 0.1


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    configure_store(str(tmp_path / "ledger.db"))
    yield
    configure_store(None)


def _plan():
    return client.post("/dispatch", json=DISPATCH).json()


def _create_complete():
    sid = client.post(
        "/dispatch/sessions",
        json={"dispatch": DISPATCH, "plan_snapshot": _plan(), "audit_tolerance_kwh": TOL},
    ).json()["session_id"]
    assert client.post(
        f"/dispatch/sessions/{sid}/batches",
        json={"start_index": 0, "records": RECORDS[:2]},
    ).status_code == 200
    assert client.post(
        f"/dispatch/sessions/{sid}/batches",
        json={"start_index": 2, "records": RECORDS[2:]},
    ).status_code == 200
    return sid


def _rev(sid, expected, changes):
    return client.post(
        f"/dispatch/sessions/{sid}/revisions",
        json={"expected_revision": expected, "changes": changes},
    )


def _issue(sid, revision):
    return client.post(
        f"/dispatch/sessions/{sid}/vouchers",
        json={"expected_revision": revision},
    )


def _ch(index, record):
    return {"index": index, "record": record}


def _post(account, items):
    return client.post(
        f"/accounts/{account}/ledger",
        json={"items": [{"session_id": s, "voucher_no": n} for s, n in items]},
    )


def _get(account):
    return client.get(f"/accounts/{account}/ledger")


# ---------------------------------------------------------------------------
# 计费栏方向：首单 +、应补 +、应退 −、零调整 0；余额可负
# ---------------------------------------------------------------------------


def test_first_voucher_full_amount_increases_net_receivable():
    sid = _create_complete()
    v = _issue(sid, 0).json()

    r = _post("acct-1", [(sid, 1)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["result"] == "posted"
    assert body["posted_count"] == 1
    assert body["batch_net_change_yuan"] == pytest.approx(ORIG_TOTAL, abs=1e-9)
    assert body["balance_yuan"] == pytest.approx(ORIG_TOTAL, abs=1e-9)
    assert body["entry_count"] == 1
    e = body["entries"][0]
    assert e["entry_no"] == 1
    assert e["session_id"] == sid
    assert e["voucher_no"] == 1
    assert e["voucher_id"] == v["voucher_id"]
    assert e["revision_no"] == 0
    assert e["billing"]["type"] == "full"
    assert e["amount_yuan"] == pytest.approx(ORIG_TOTAL, abs=1e-9)
    assert e["balance_after_yuan"] == pytest.approx(ORIG_TOTAL, abs=1e-9)
    assert e["total_cost_yuan"] == pytest.approx(ORIG_TOTAL, abs=1e-9)
    assert e["posted_at"].endswith("Z")


def test_supplement_and_refund_directions_in_one_batch():
    sid = _create_complete()
    # 凭证 1：全额 0.10（rev0）
    _issue(sid, 0)
    # 凭证 2：应补 0.01（rev1，t1 购电 0.02）
    up1 = dict(RECORDS[1]); up1.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    _rev(sid, 0, [_ch(1, up1)])
    v2 = _issue(sid, 1).json()
    assert v2["billing"]["type"] == "supplement"
    # 凭证 3：应退 0.02（跳过 rev2，对 rev3 出单，总费用 0.09）
    up2 = dict(RECORDS[1]); up2.update(base_load_kw=0.24, grid_purchase_energy_kwh=0.03)
    _rev(sid, 1, [_ch(1, up2)])
    down = dict(RECORDS[1]); down.update(base_load_kw=0.18, grid_purchase_energy_kwh=0.0)
    _rev(sid, 2, [_ch(1, down)])
    v3 = _issue(sid, 3).json()
    assert v3["billing"]["type"] == "refund"
    assert v3["billing"]["amount_yuan"] == pytest.approx(0.02, abs=1e-9)

    # 同批按序把 1、2、3 一次入账
    r = _post("acct-a", [(sid, 1), (sid, 2), (sid, 3)])
    assert r.status_code == 200, r.text
    body = r.json()
    amounts = [e["amount_yuan"] for e in body["entries"]]
    assert amounts == pytest.approx([0.10, 0.01, -0.02], abs=1e-9)
    # 本批净变动 +0.09，余额 0.09
    assert body["batch_net_change_yuan"] == pytest.approx(0.09, abs=1e-9)
    assert body["balance_yuan"] == pytest.approx(0.09, abs=1e-9)
    assert [e["balance_after_yuan"] for e in body["entries"]] == pytest.approx(
        [0.10, 0.11, 0.09], abs=1e-9
    )
    assert [e["entry_no"] for e in body["entries"]] == [1, 2, 3]
    # 计费栏相对坐标随冻结凭证一并落账（首单为 null，其后指向上一张凭证）
    assert body["entries"][0]["billing"]["relative_to_voucher_no"] is None
    assert body["entries"][1]["billing"]["relative_to_revision"] == 0
    assert body["entries"][1]["billing"]["relative_to_voucher_no"] == 1
    assert body["entries"][2]["billing"]["relative_to_revision"] == 1
    assert body["entries"][2]["billing"]["relative_to_voucher_no"] == 2



def test_refund_telescopes_to_latest_voucher_total():
    """同会话连续入账净额 = 最新已入凭证实测总费用（计费栏望远镜求和）。

    full +0.10 → supplement +0.10 → refund −0.10，最终余额等于凭证 3 的总费用 0.10；
    应退为负向入账，余额快照递减，接口不设非负限制。
    """
    sid = _create_complete()
    _issue(sid, 0)  # 全额 0.10
    # rev1：t1、t2 负荷 0.30kW、放电 0.09kWh 不变 → 期望购电 0.5*0.30-0.09=0.06，
    # 每段多购 0.05kWh（峰时 1 元）→ 总费用 0.20，凭证 2 应补 0.10
    n1 = dict(RECORDS[1]); n1.update(base_load_kw=0.30, grid_purchase_energy_kwh=0.06)
    n2 = dict(RECORDS[2]); n2.update(base_load_kw=0.30, grid_purchase_energy_kwh=0.06)
    assert _rev(sid, 0, [_ch(1, n1), _ch(2, n2)]).status_code == 200
    v2 = _issue(sid, 1).json()
    assert v2["billing"]["type"] == "supplement"
    assert v2["billing"]["amount_yuan"] == pytest.approx(0.10, abs=1e-9)
    # rev2：回到最初台账（总费用 0.10）→ 凭证 3 应退 0.10
    assert _rev(sid, 1, [_ch(1, RECORDS[1]), _ch(2, RECORDS[2])]).status_code == 200
    v3 = _issue(sid, 2).json()
    assert v3["billing"]["type"] == "refund"
    assert v3["billing"]["amount_yuan"] == pytest.approx(0.10, abs=1e-9)

    r = _post("tel", [(sid, 1), (sid, 2), (sid, 3)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert [e["amount_yuan"] for e in body["entries"]] == pytest.approx(
        [0.10, 0.10, -0.10], abs=1e-9
    )
    assert [e["balance_after_yuan"] for e in body["entries"]] == pytest.approx(
        [0.10, 0.20, 0.10], abs=1e-9
    )
    # 望远镜求和：同会话净额恒等于最新已入凭证的实测总费用（≥0，因费用本身非负）
    assert body["batch_net_change_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert body["balance_yuan"] == pytest.approx(0.10, abs=1e-9)


def test_balance_column_allows_negative():
    """承诺「余额允许为负」：台账余额列无非负约束，负向快照可正常落盘读回。

    正常业务流中单个会话的入账净额望远镜求和为最新凭证的非负总费用，负余额不会
    自然产生；此处保证存储层不设 CHECK 拦截，将来数据源扩展时也不会误拒负余额。
    """
    from app.storage import get_store

    sid = _create_complete()
    v = _issue(sid, 0).json()
    store = get_store()
    with store.write_transaction() as conn:
        store.insert_ledger_entry(
            conn,
            entry_id="neg-1", account_id="minus", entry_no=1,
            session_id=sid, voucher_no=1, voucher_id=v["voucher_id"],
            revision_no=0, billing_type="refund", voucher_amount_yuan=0.25,
            relative_to_revision=0, relative_to_voucher_no=1,
            signed_amount_yuan=-0.25, balance_after_yuan=-0.25,
            total_cost_yuan=0.10, posted_at="2026-01-01T00:00:00Z",
        )
    detail = _get("minus").json()
    assert detail["balance_yuan"] == pytest.approx(-0.25, abs=1e-9)
    assert detail["entries"][0]["amount_yuan"] == pytest.approx(-0.25, abs=1e-9)
    assert detail["entries"][0]["balance_after_yuan"] == pytest.approx(-0.25, abs=1e-9)

    # 该 (session, voucher) 已占主键：经正常接口记到别的账户按跨账户拒绝
    r = _post("other", [(sid, 1)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_voucher_cross_account"


def test_no_adjustment_records_zero_entry_and_keeps_balance():
    """no_adjustment 记一条 0 元明细，余额不变但仍占入账序号。"""
    sid = _create_complete()
    _issue(sid, 0)
    # rev1 只改 t3 谷时购电量（价格 0）→ 总费用不变 → no_adjustment
    n3 = dict(RECORDS[3]); n3.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.11)
    _rev(sid, 0, [_ch(3, n3)])
    v2 = _issue(sid, 1).json()
    assert v2["billing"]["type"] == "no_adjustment"

    r = _post("z", [(sid, 1), (sid, 2)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert [e["amount_yuan"] for e in body["entries"]] == [0.10, 0.0]
    assert body["batch_net_change_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert body["balance_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert body["entry_count"] == 2
    assert [e["billing"]["type"] for e in body["entries"]] == ["full", "no_adjustment"]


# ---------------------------------------------------------------------------
# 连续入账：同批补齐前序；跳号整批拒绝
# ---------------------------------------------------------------------------


def test_must_start_from_first_voucher():
    sid = _create_complete()
    _issue(sid, 0)
    r = _post("a", [(sid, 2)])  # 没有凭证 2，且跳过 1
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "ledger_voucher_not_found"
    assert body["session_id"] == sid and body["voucher_no"] == 2
    assert _get("a").json()["entry_count"] == 0


def test_gap_after_first_voucher_rejected():
    sid = _create_complete()
    for rev in range(3):  # 出三张凭证
        if rev == 0:
            _issue(sid, 0)
        else:
            n3 = dict(RECORDS[3]); n3.update(base_load_kw=0.2 + 0.02 * rev,
                                             grid_purchase_energy_kwh=0.1 + 0.01 * rev)
            _rev(sid, rev - 1, [_ch(3, n3)])
            _issue(sid, rev)
    assert _post("a", [(sid, 1)]).status_code == 200
    # 跳到 3（缺 2）
    r = _post("a", [(sid, 3)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_voucher_gap"
    assert r.json()["expected_voucher_no"] == 2
    # 同批带 [3, 2] 顺序颠倒也不行；按序 [2, 3] 补齐成功
    r = _post("a", [(sid, 3), (sid, 2)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_voucher_gap"
    ok = _post("a", [(sid, 2), (sid, 3)])
    assert ok.status_code == 200, ok.text
    assert [e["voucher_no"] for e in ok.json()["entries"]] == [2, 3]
    assert _get("a").json()["entry_count"] == 3


def test_unissued_voucher_no_rejected():
    sid = _create_complete()
    _issue(sid, 0)
    r = _post("a", [(sid, 1), (sid, 2)])  # 凭证 2 从未出具
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_voucher_not_found"
    assert r.json()["voucher_no"] == 2
    # 整批原子：凭证 1 也不落盘
    assert _get("a").json()["entries"] == []


def test_unknown_session_rejected():
    r = _post("a", [("no-such-session", 1)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_voucher_not_found"


# ---------------------------------------------------------------------------
# 重复与跨账户
# ---------------------------------------------------------------------------


def test_duplicate_within_batch_rejected_by_request_validation():
    sid = _create_complete()
    _issue(sid, 0)
    r = client.post(
        "/accounts/a/ledger",
        json={"items": [
            {"session_id": sid, "voucher_no": 1},
            {"session_id": sid, "voucher_no": 1},
        ]},
    )
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_input"


def test_duplicate_history_rejected_entire_batch():
    sid = _create_complete()
    _issue(sid, 0)
    assert _post("a", [(sid, 1)]).status_code == 200
    sid2 = _create_complete()
    _issue(sid2, 0)
    # 第二批里混入已入账凭证 + 新会话首单 → 整批拒绝，sid2 也不落盘
    r = _post("a", [(sid2, 1), (sid, 1)])
    assert r.status_code == 409
    body = r.json()
    assert body["error"] == "ledger_voucher_already_posted"
    assert body["entry_no"] == 1
    detail = _get("a").json()
    assert detail["entry_count"] == 1
    assert [e["session_id"] for e in detail["entries"]] == [sid]
    assert detail["balance_yuan"] == pytest.approx(0.10, abs=1e-9)


def test_cross_account_rejected():
    sid = _create_complete()
    _issue(sid, 0)
    assert _post("acct-left", [(sid, 1)]).status_code == 200
    r = _post("acct-right", [(sid, 1)])
    assert r.status_code == 422
    body = r.json()
    assert body["error"] == "ledger_voucher_cross_account"
    assert body["account_id"] == "acct-left"
    # 右账户余额不受影响
    assert _get("acct-right").json()["balance_yuan"] == 0.0
    # 跨账户拒绝不影响原账户：出具新凭证后仍可在原账户接着连续入账
    n3 = dict(RECORDS[3]); n3.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.11)
    assert _rev(sid, 0, [_ch(3, n3)]).status_code == 200
    assert _issue(sid, 1).status_code == 200
    ok = _post("acct-left", [(sid, 2)])
    assert ok.status_code == 200, ok.text
    assert ok.json()["entry_count"] == 2


def test_cross_account_mid_batch_leaves_nothing():
    """批内第一张是别的账户已记的凭证：整批拒绝，同批其余凭证也不落盘。"""
    sid = _create_complete()
    _issue(sid, 0)
    _post("owner", [(sid, 1)])
    sid2 = _create_complete()
    _issue(sid2, 0)
    r = _post("intruder", [(sid2, 1), (sid, 1)])
    assert r.status_code == 422
    assert r.json()["error"] == "ledger_voucher_cross_account"
    assert _get("intruder").json()["entry_count"] == 0


# ---------------------------------------------------------------------------
# 多会话汇聚 + 查询次序
# ---------------------------------------------------------------------------


def test_multiple_sessions_aggregate_in_post_order():
    s1 = _create_complete()
    s2 = _create_complete()
    _issue(s1, 0)
    _issue(s2, 0)
    up = dict(RECORDS[1]); up.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    _rev(s1, 0, [_ch(1, up)])
    _issue(s1, 1)  # s1 凭证 2：应补 0.01

    r = _post("family", [(s1, 1), (s2, 1), (s1, 2)])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["posted_count"] == 3
    assert body["batch_net_change_yuan"] == pytest.approx(0.21, abs=1e-9)
    assert body["balance_yuan"] == pytest.approx(0.21, abs=1e-9)
    assert [e["entry_no"] for e in body["entries"]] == [1, 2, 3]
    assert [(e["session_id"], e["voucher_no"]) for e in body["entries"]] == [
        (s1, 1), (s2, 1), (s1, 2)
    ]

    detail = _get("family").json()
    assert detail["entry_count"] == 3
    assert detail["balance_yuan"] == pytest.approx(0.21, abs=1e-9)
    assert [e["entry_no"] for e in detail["entries"]] == [1, 2, 3]
    # 每条余额快照
    assert [e["balance_after_yuan"] for e in detail["entries"]] == pytest.approx(
        [0.10, 0.20, 0.21], abs=1e-9
    )


def test_empty_account_returns_zero_balance():
    body = _get("never-used").json()
    assert body == {
        "account_id": "never-used",
        "entry_count": 0,
        "balance_yuan": 0.0,
        "entries": [],
    }


def test_empty_batch_invalid_input():
    r = client.post("/accounts/a/ledger", json={"items": []})
    assert r.status_code == 422
    assert r.json()["error"] == "invalid_input"


# ---------------------------------------------------------------------------
# 并发：不重复计费
# ---------------------------------------------------------------------------


def test_concurrent_posts_same_voucher_only_one_wins():
    sid = _create_complete()
    _issue(sid, 0)
    outcomes = []

    def worker(tag):
        try:
            res = post_vouchers("race", [{"session_id": sid, "voucher_no": 1}])
            outcomes.append((tag, "posted", res["entries"][0]["entry_no"]))
        except Exception as exc:  # noqa: BLE001
            outcomes.append((tag, getattr(exc, "code", "error"), None))

    threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    posted = [o for o in outcomes if o[1] == "posted"]
    rejected = [o for o in outcomes if o[1] == "ledger_voucher_already_posted"]
    assert len(posted) == 1
    assert len(rejected) == 9
    detail = _get("race").json()
    assert detail["entry_count"] == 1
    assert detail["balance_yuan"] == pytest.approx(0.10, abs=1e-9)


def test_concurrent_posts_distinct_vouchers_serialize_with_consistent_balance():
    """并发把不同会话的首单入到同一账户：全部成功，序号 1..N、余额为全额之和。"""
    sids = []
    for _ in range(6):
        sid = _create_complete()
        _issue(sid, 0)
        sids.append(sid)
    errors = []

    def worker(sid):
        try:
            post_vouchers("family", [{"session_id": sid, "voucher_no": 1}])
        except Exception as exc:  # noqa: BLE001
            errors.append(getattr(exc, "code", str(exc)))

    threads = [threading.Thread(target=worker, args=(sid,)) for sid in sids]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    detail = _get("family").json()
    assert detail["entry_count"] == 6
    assert sorted(e["session_id"] for e in detail["entries"]) == sorted(sids)
    assert [e["entry_no"] for e in detail["entries"]] == [1, 2, 3, 4, 5, 6]
    assert detail["balance_yuan"] == pytest.approx(0.60, abs=1e-9)
    # entry_no 与余额快照严格递增一致
    for i, e in enumerate(detail["entries"], start=1):
        assert e["entry_no"] == i
        assert e["balance_after_yuan"] == pytest.approx(0.10 * i, abs=1e-9)


# ---------------------------------------------------------------------------
# 更正 / 重试出单不动旧入账；新凭证可续入
# ---------------------------------------------------------------------------


def test_revision_and_reissue_do_not_change_posted_entries():
    sid = _create_complete()
    v1 = _issue(sid, 0).json()
    assert _post("a", [(sid, 1)]).status_code == 200

    # 重试出单返回冻结原凭证
    again = _issue(sid, 0).json()
    assert again["result"] == "existing"
    assert again["voucher_id"] == v1["voucher_id"]

    # 更正后出新凭证并续入：旧明细不变，余额累加新计费栏
    up = dict(RECORDS[1]); up.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(sid, 0, [_ch(1, up)]).status_code == 200
    v2 = _issue(sid, 1).json()
    assert _post("a", [(sid, 2)]).status_code == 200

    detail = _get("a").json()
    assert detail["entry_count"] == 2
    first = detail["entries"][0]
    assert first["voucher_id"] == v1["voucher_id"]
    assert first["revision_no"] == 0
    assert first["amount_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert first["total_cost_yuan"] == pytest.approx(0.10, abs=1e-9)
    assert detail["entries"][1]["voucher_id"] == v2["voucher_id"]
    assert detail["balance_yuan"] == pytest.approx(0.11, abs=1e-9)


# ---------------------------------------------------------------------------
# 持久化：重启后明细与余额仍在
# ---------------------------------------------------------------------------


def test_ledger_persists_across_restart(tmp_path):
    db_path = str(tmp_path / "persist-ledger.db")
    configure_store(db_path)
    s1 = _create_complete()
    s2 = _create_complete()
    _issue(s1, 0)
    _issue(s2, 0)
    _post("fam", [(s1, 1), (s2, 1)])

    configure_store(db_path)  # 模拟进程重启
    detail = _get("fam").json()
    assert detail["entry_count"] == 2
    assert detail["balance_yuan"] == pytest.approx(0.20, abs=1e-9)
    assert [e["session_id"] for e in detail["entries"]] == [s1, s2]

    # 重启后重复入账被拒；新凭证可续入
    dup = _post("fam", [(s1, 1)])
    assert dup.status_code == 409
    up = dict(RECORDS[1]); up.update(base_load_kw=0.22, grid_purchase_energy_kwh=0.02)
    assert _rev(s1, 0, [_ch(1, up)]).status_code == 200
    _issue(s1, 1)
    cont = _post("fam", [(s1, 2)])
    assert cont.status_code == 200, cont.text
    assert cont.json()["entry_count"] == 3
    assert cont.json()["balance_yuan"] == pytest.approx(0.21, abs=1e-9)


def test_legacy_db_ledger_table_auto_created(tmp_path):
    """旧库（只有 sessions / session_records）打开即补建台账表，空账户可查、可入账。"""
    import json
    import sqlite3

    db_path = str(tmp_path / "legacy-ledger.db")
    plan = client.post("/dispatch", json=DISPATCH).json()
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE sessions (
            session_id TEXT PRIMARY KEY,
            dispatch_json TEXT NOT NULL,
            plan_snapshot_json TEXT NOT NULL,
            audit_tolerance_kwh REAL NOT NULL,
            num_periods INTEGER NOT NULL,
            created_at TEXT NOT NULL
        );
        CREATE TABLE session_records (
            session_id TEXT NOT NULL,
            idx INTEGER NOT NULL,
            record_json TEXT NOT NULL,
            stored_at TEXT NOT NULL,
            PRIMARY KEY (session_id, idx)
        );
        """
    )
    sid = "legacy-ledger-session"
    conn.execute(
        "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
        (sid, json.dumps(DISPATCH), json.dumps(plan), TOL, 4, "2026-01-01T00:00:00Z"),
    )
    for idx, rec in enumerate(RECORDS):
        conn.execute(
            "INSERT INTO session_records VALUES (?, ?, ?, ?)",
            (sid, idx, json.dumps(rec), "2026-01-01T00:00:01Z"),
        )
    conn.commit()
    conn.close()

    configure_store(db_path)
    assert _get("new-acct").json()["entry_count"] == 0
    # 该旧库无凭证表 → 出单后再入账端到端可用
    r = _issue(sid, 0)
    assert r.status_code == 200, r.text
    posted = _post("new-acct", [(sid, 1)])
    assert posted.status_code == 200, posted.text
    assert posted.json()["balance_yuan"] == pytest.approx(ORIG_TOTAL, abs=1e-9)


def test_ledger_table_added_columns_migrated(tmp_path):
    """已建台账表（缺后加的相对凭证坐标列）打开即 ALTER 补列，读写正常。"""
    import sqlite3

    db_path = str(tmp_path / "migrate-ledger.db")
    conn = sqlite3.connect(db_path)
    # 模拟后加列之前的台账表结构（无 relative_to_* 两列）
    conn.execute(
        """
        CREATE TABLE account_ledger_entries (
            entry_id TEXT NOT NULL, account_id TEXT NOT NULL, entry_no INTEGER NOT NULL,
            session_id TEXT NOT NULL, voucher_no INTEGER NOT NULL, voucher_id TEXT NOT NULL,
            revision_no INTEGER NOT NULL, billing_type TEXT NOT NULL,
            voucher_amount_yuan REAL NOT NULL, signed_amount_yuan REAL NOT NULL,
            balance_after_yuan REAL NOT NULL, total_cost_yuan REAL NOT NULL,
            posted_at TEXT NOT NULL,
            PRIMARY KEY (session_id, voucher_no), UNIQUE (account_id, entry_no)
        )
        """
    )
    conn.commit()
    conn.close()

    configure_store(db_path)  # 打开即迁移
    sid = _create_complete()
    _issue(sid, 0)
    r = _post("mig", [(sid, 1)])
    assert r.status_code == 200, r.text
    e = r.json()["entries"][0]
    # 旧行结构下首单相对坐标为 null，新写入路径不再报错
    assert e["billing"]["type"] == "full"
    assert e["billing"]["relative_to_voucher_no"] is None
    assert _get("mig").json()["entry_count"] == 1
