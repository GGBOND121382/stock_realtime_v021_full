"use strict";

var queueLib = require("./lib/manual_order_queue.js");

var CONFIG_NAME = "config.json";
var DEFAULT_CONFIG = {
  csv_file: "orders.csv",
  target_seconds: 120,
  allow_duplicates: false,
  show_price: true,
  floaty_x: 8,
  floaty_y: 180,
  debounce_ms: 350
};

var windowRef = null;
var orders = [];
var state = null;
var config = null;
var statePath = null;
var currentIndex = -1;
var lastActionAt = 0;
var dragState = null;

function mergeConfig(base, extra) {
  var out = {};
  var key;
  for (key in base) if (Object.prototype.hasOwnProperty.call(base, key)) out[key] = base[key];
  for (key in extra) if (Object.prototype.hasOwnProperty.call(extra, key)) out[key] = extra[key];
  return out;
}

function loadConfig() {
  var path = files.join(files.cwd(), CONFIG_NAME);
  if (!files.exists(path)) return mergeConfig(DEFAULT_CONFIG, {});
  var raw = JSON.parse(files.read(path));
  return mergeConfig(DEFAULT_CONFIG, raw || {});
}

function readPlan(cfg) {
  var path = files.join(files.cwd(), String(cfg.csv_file || "orders.csv"));
  if (!files.exists(path)) throw new Error("missing CSV: " + path);
  var text = files.read(path);
  return {
    path: path,
    text: text,
    fingerprint: queueLib.fnv1a(text),
    orders: queueLib.parseOrdersCsv(text, { allow_duplicates: cfg.allow_duplicates === true })
  };
}

function loadState(fingerprint, planOrders) {
  statePath = files.join(files.cwd(), "manual_assistant_state_" + fingerprint + ".json");
  var raw = null;
  if (files.exists(statePath)) {
    try { raw = JSON.parse(files.read(statePath)); } catch (e) { raw = null; }
  }
  return queueLib.sanitizeState(raw, fingerprint, planOrders);
}

function saveState() {
  if (!statePath || !state) return;
  state.updated_at = new Date().toISOString();
  files.write(statePath, JSON.stringify(state, null, 2));
}

function sideLabel(side) {
  return side === "sell" ? "卖出" : "买入";
}

function formatElapsed(ms) {
  var sec = Math.max(0, Math.floor(ms / 1000));
  var m = Math.floor(sec / 60);
  var s = sec % 60;
  return (m < 10 ? "0" : "") + m + ":" + (s < 10 ? "0" : "") + s;
}

function runUi(fn) {
  try { ui.run(fn); } catch (e) { fn(); }
}

function currentOrder() {
  return currentIndex >= 0 && currentIndex < orders.length ? orders[currentIndex] : null;
}

function nextIndexForPhase(start) {
  return queueLib.findNextIndex(state, orders, state.phase, start);
}

function setCurrent(index) {
  currentIndex = index;
  if (state.phase === "skipped") state.review_cursor = index < 0 ? state.review_cursor : index;
  else state.cursor = index < 0 ? state.cursor : index;
  saveState();
  render();
}

function advance() {
  var start = currentIndex < 0 ? 0 : currentIndex + 1;
  var next = nextIndexForPhase(start);
  if (next < 0 && state.phase === "normal") {
    var c = queueLib.counts(state, orders);
    if (c.skipped > 0) {
      state.phase = "skipped";
      state.review_cursor = 0;
      saveState();
      next = nextIndexForPhase(0);
    }
  }
  if (next < 0) {
    state.completed_at = new Date().toISOString();
    saveState();
  }
  setCurrent(next);
}

function debounce() {
  var now = Date.now();
  var wait = Number(config.debounce_ms || 350);
  if (now - lastActionAt < wait) return false;
  lastActionAt = now;
  return true;
}

function copyText(kind, value) {
  if (!debounce()) return;
  setClip(String(value));
  var order = currentOrder();
  if (order) {
    state.history.push({ at: new Date().toISOString(), order_id: order.id, action: "copy_" + kind });
    if (state.history.length > 200) state.history = state.history.slice(state.history.length - 200);
    saveState();
  }
  toast("已复制 " + String(value));
}

function markDone() {
  if (!debounce()) return;
  var order = currentOrder();
  if (!order) return;
  queueLib.markDone(state, order);
  saveState();
  advance();
}

function markSkipped() {
  if (!debounce()) return;
  var order = currentOrder();
  if (!order) return;
  if (state.phase === "skipped") {
    state.history.push({ at: new Date().toISOString(), order_id: order.id, action: "defer_skipped" });
    saveState();
    var next = queueLib.findNextIndex(state, orders, "skipped", currentIndex + 1);
    if (next === currentIndex) next = -1;
    if (next < 0) {
      state.completed_at = new Date().toISOString();
      saveState();
    }
    setCurrent(next);
    return;
  }
  queueLib.markSkipped(state, order);
  saveState();
  advance();
}

function reopenPrevious() {
  if (!debounce()) return;
  if (!state.history || !state.history.length) {
    toast("没有可回退记录");
    return;
  }
  for (var i = state.history.length - 1; i >= 0; i--) {
    var entry = state.history[i];
    if (entry.action !== "done" && entry.action !== "skipped") continue;
    for (var j = 0; j < orders.length; j++) {
      if (orders[j].id === entry.order_id) {
        queueLib.reopen(state, orders[j]);
        state.phase = "normal";
        saveState();
        setCurrent(j);
        return;
      }
    }
  }
  toast("没有可回退记录");
}

function render() {
  if (!windowRef) return;
  var c = queueLib.counts(state, orders);
  var order = currentOrder();
  var started = state.started_at ? new Date(state.started_at).getTime() : Date.now();
  var elapsed = Date.now() - started;
  var targetMs = Math.max(1, Number(config.target_seconds || 120)) * 1000;
  var expectedDone = Math.min(c.total, Math.floor((elapsed / targetMs) * c.total));
  var paceDelta = c.done - expectedDone;
  var phaseText = state.phase === "skipped" ? "异常补单" : "正常队列";

  runUi(function () {
    windowRef.progress.setText("" + c.done + "/" + c.total + "  跳过 " + c.skipped + "  " + phaseText);
    windowRef.timer.setText(formatElapsed(elapsed) + " / " + formatElapsed(targetMs) + "  节奏 " + (paceDelta >= 0 ? "+" : "") + paceDelta);

    if (!order) {
      windowRef.side.setText("本轮完成");
      windowRef.code.setText("—");
      windowRef.name.setText(c.skipped > 0 ? "仍有 " + c.skipped + " 笔保留在异常队列" : "全部已人工处理");
      windowRef.qty.setText("数量 —");
      windowRef.price.setText("参考价 —");
      windowRef.copyCode.setEnabled(false);
      windowRef.copyQty.setEnabled(false);
      windowRef.copyPrice.setEnabled(false);
      windowRef.done.setEnabled(false);
      windowRef.skip.setEnabled(false);
      return;
    }

    windowRef.side.setText(sideLabel(order.side) + "  第 " + order.sequence + " 笔");
    windowRef.code.setText(order.code + (order.market ? "." + order.market : ""));
    windowRef.name.setText(order.name ? order.name : "请在同花顺确认证券名称");
    windowRef.qty.setText("数量 " + order.qty + " 股");
    windowRef.price.setText(config.show_price !== false && order.price_text ? "CSV参考价 " + order.price_text : "参考价未显示");
    windowRef.copyCode.setEnabled(true);
    windowRef.copyQty.setEnabled(true);
    windowRef.copyPrice.setEnabled(config.show_price !== false && !!order.price_text);
    windowRef.done.setEnabled(true);
    windowRef.skip.setEnabled(true);
    windowRef.skip.setText(state.phase === "skipped" ? "仍异常 →" : "跳过 →");
  });
}

function attachDragHandle() {
  try {
    windowRef.drag.setOnTouchListener(function (view, event) {
      var action = event.getAction();
      if (action === event.ACTION_DOWN) {
        dragState = {
          rawX: event.getRawX(),
          rawY: event.getRawY(),
          winX: windowRef.getX(),
          winY: windowRef.getY(),
          moved: false
        };
        return true;
      }
      if (action === event.ACTION_MOVE && dragState) {
        var dx = event.getRawX() - dragState.rawX;
        var dy = event.getRawY() - dragState.rawY;
        if (Math.abs(dx) + Math.abs(dy) > 8) dragState.moved = true;
        windowRef.setPosition(dragState.winX + dx, dragState.winY + dy);
        return true;
      }
      if (action === event.ACTION_UP) {
        dragState = null;
        return true;
      }
      return false;
    });
  } catch (e) {
    console.warn("drag handle unavailable: " + e);
  }
}

function buildWindow() {
  try {
    windowRef = floaty.window(
      <vertical bg="#EFFFFFFF" padding="8" w="310">
        <text id="drag" text="AS1455 人工下单助手  ·  拖动这里" textSize="14sp" textStyle="bold" gravity="center" h="36" />
        <text id="progress" text="0/0" textSize="14sp" gravity="center" />
        <text id="timer" text="00:00 / 02:00" textSize="13sp" gravity="center" />
        <text id="side" text="—" textSize="18sp" textStyle="bold" gravity="center" marginTop="5" />
        <text id="code" text="000000" textSize="30sp" textStyle="bold" gravity="center" />
        <text id="name" text="证券名称" textSize="13sp" gravity="center" />
        <text id="qty" text="数量 —" textSize="24sp" textStyle="bold" gravity="center" marginTop="5" />
        <text id="price" text="CSV参考价 —" textSize="16sp" gravity="center" />
        <horizontal marginTop="7">
          <button id="copyCode" text="复制代码" w="*" h="48" />
          <button id="copyQty" text="复制数量" w="*" h="48" />
          <button id="copyPrice" text="复制价格" w="*" h="48" />
        </horizontal>
        <horizontal marginTop="5">
          <button id="done" text="已下单 →" w="*" h="54" />
          <button id="skip" text="跳过 →" w="*" h="54" />
        </horizontal>
        <horizontal marginTop="4">
          <button id="back" text="回退上一笔" w="*" h="42" />
          <button id="close" text="关闭助手" w="*" h="42" />
        </horizontal>
      </vertical>
    );
  } catch (e) {
    throw new Error("无法创建悬浮窗。请先授予 AutoJs6 悬浮窗权限: " + e);
  }

  windowRef.setPosition(Number(config.floaty_x || 8), Number(config.floaty_y || 180));
  attachDragHandle();

  windowRef.copyCode.click(function () {
    var o = currentOrder();
    if (o) copyText("code", o.code);
  });
  windowRef.copyQty.click(function () {
    var o = currentOrder();
    if (o) copyText("qty", o.qty);
  });
  windowRef.copyPrice.click(function () {
    var o = currentOrder();
    if (o && o.price_text) copyText("price", o.price_text);
  });
  windowRef.done.click(markDone);
  windowRef.skip.click(markSkipped);
  windowRef.back.click(reopenPrevious);
  windowRef.close.click(function () {
    saveState();
    if (windowRef) windowRef.close();
    windowRef = null;
    exit();
  });
}

function preflightSummary(plan) {
  var buys = plan.orders.filter(function (o) { return o.side === "buy"; }).length;
  var sells = plan.orders.length - buys;
  return [
    "订单文件: " + config.csv_file,
    "订单数: " + plan.orders.length + "  买 " + buys + " / 卖 " + sells,
    "目标节奏: " + config.target_seconds + " 秒",
    "",
    "开始前请确认同花顺：",
    "1. 已登录正确券商/账户；",
    "2. 当前委托类型和价格口径正确；",
    "3. 横竖屏与页面布局已固定；",
    "4. 屏幕不会在执行中锁屏；",
    "5. 最终委托由你本人核对并确认。",
    "",
    "本助手不会读取、点击或控制同花顺，只显示订单、复制字段并记录人工进度。"
  ].join("\n");
}

function start() {
  config = loadConfig();
  var plan = readPlan(config);
  orders = plan.orders;
  if (!orders.length) throw new Error("CSV contains no orders");
  state = loadState(plan.fingerprint, orders);

  if (!dialogs.confirm("AS1455 高速人工下单助手", preflightSummary(plan))) return;
  if (!state.started_at || queueLib.counts(state, orders).done === 0) state.started_at = new Date().toISOString();
  state.completed_at = null;
  saveState();

  buildWindow();
  currentIndex = nextIndexForPhase(state.phase === "skipped" ? state.review_cursor : state.cursor);
  if (currentIndex < 0 && state.phase === "normal" && queueLib.counts(state, orders).skipped > 0) {
    state.phase = "skipped";
    currentIndex = nextIndexForPhase(0);
  }
  render();

  events.on("exit", function () {
    try { saveState(); } catch (e) {}
    try { if (windowRef) windowRef.close(); } catch (e2) {}
    try { floaty.closeAll(); } catch (e3) {}
  });

  setInterval(render, 250);
}

try {
  start();
} catch (e) {
  console.error(e && e.stack ? e.stack : String(e));
  dialogs.alert("AS1455 人工助手启动失败", String(e));
}
