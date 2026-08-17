"use strict";

auto.waitFor();

var THS_PACKAGE = "com.hexin.plat.android";
var WAIT_SECONDS = 15;
var MAX_NODES = 300;

var knownIds = [
  "com.hexin.plat.android:id/btn",
  "com.hexin.plat.android:id/auto_stockcode",
  "com.hexin.plat.android:id/dialogplus_view_container",
  "com.hexin.plat.android:id/stockcode_tv",
  "com.hexin.plat.android:id/stockvolume",
  "com.hexin.plat.android:id/stockprice",
  "com.hexin.plat.android:id/btn_transaction",
  "com.hexin.plat.android:id/ok_btn",
  "com.hexin.plat.android:id/cancel_btn",
  "com.hexin.plat.android:id/prompt_content"
];

function safeValue(fn, fallback) {
  try {
    var value = fn();
    return value === null || value === undefined ? fallback : value;
  } catch (e) {
    return fallback;
  }
}

function clean(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/\r/g, "\\r")
    .replace(/\n/g, "\\n")
    .replace(/\t/g, "\\t");
}

function nodeLine(node, index) {
  var rid = safeValue(function () { return node.id(); }, "");
  var text = safeValue(function () { return node.text(); }, "");
  var desc = safeValue(function () { return node.desc(); }, "");
  var cls = safeValue(function () { return node.className(); }, "");
  var clickable = safeValue(function () { return node.clickable(); }, false);
  var enabled = safeValue(function () { return node.enabled(); }, false);
  var bounds = safeValue(function () { return node.bounds(); }, null);
  var boundsText = bounds ? clean(bounds.toString()) : "";
  return "[" + index + "]" +
    " id=" + clean(rid) +
    " text=" + JSON.stringify(clean(text)) +
    " desc=" + JSON.stringify(clean(desc)) +
    " class=" + clean(cls) +
    " clickable=" + clickable +
    " enabled=" + enabled +
    " bounds=" + boundsText;
}

function waitForThs() {
  toast("请在 " + WAIT_SECONDS + " 秒内切到同花顺，并停留在买入或卖出页面");
  for (var left = WAIT_SECONDS; left > 0; left--) {
    if (currentPackage() === THS_PACKAGE) return true;
    if (left === WAIT_SECONDS || left === 10 || left === 5) {
      toast("等待同花顺页面：还剩 " + left + " 秒");
    }
    sleep(1000);
  }
  return currentPackage() === THS_PACKAGE;
}

if (!waitForThs()) {
  console.show();
  console.error("[FAIL] 当前包不是同花顺: " + currentPackage());
  console.error("请重新运行脚本，并在倒计时内切到同花顺买入/卖出页面。脚本不会主动启动或跳转同花顺。");
  dialogs.alert("探测失败", "没有在倒计时内检测到同花顺。请重新运行后手工切到买入或卖出页面。");
  exit();
}

// Let the target page settle before taking the accessibility snapshot.
sleep(2000);

var packageName = currentPackage();
var activityName = currentActivity();
var knownResults = [];
knownIds.forEach(function (rid) {
  var found = id(rid).find();
  knownResults.push(rid + " count=" + found.size());
});

var allNodes;
try {
  allNodes = classNameMatches(/.*/).find();
} catch (e) {
  allNodes = null;
}

var dumpLines = [];
if (allNodes) {
  var total = allNodes.size();
  var limit = Math.min(total, MAX_NODES);
  for (var i = 0; i < limit; i++) {
    var node = allNodes.get(i);
    var rid = safeValue(function () { return node.id(); }, "");
    var text = safeValue(function () { return node.text(); }, "");
    var desc = safeValue(function () { return node.desc(); }, "");
    var clickable = safeValue(function () { return node.clickable(); }, false);
    // Keep nodes that expose an identifier/user-visible content, plus clickable nodes.
    if (rid || text || desc || clickable) dumpLines.push(nodeLine(node, i));
  }
  if (total > MAX_NODES) {
    dumpLines.push("[TRUNCATED] total_nodes=" + total + " max_nodes=" + MAX_NODES);
  }
} else {
  dumpLines.push("[WARN] 无法通过 classNameMatches(/.*/) 枚举无障碍节点");
}

// Show the console only after sampling so the floating console does not affect the target snapshot.
console.show();
console.log("========== THS UI PROBE ==========");
console.log("当前包: " + packageName);
console.log("当前Activity: " + activityName);
console.log("---------- known resource ids ----------");
knownResults.forEach(function (line) { console.log(line); });
console.log("---------- accessibility nodes ----------");
if (!dumpLines.length) {
  console.log("[EMPTY] 当前页面没有枚举到带 id/text/desc/clickable 的无障碍节点");
} else {
  dumpLines.forEach(function (line) { console.log(line); });
}
console.log("========== END THS UI PROBE ==========");
toast("同花顺页面探测完成，请截图或复制控制台输出");
