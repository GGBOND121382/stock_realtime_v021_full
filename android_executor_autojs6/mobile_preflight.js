"use strict";

auto.waitFor();

var guard = require("./lib/mobile_ui_guard.js");

function loadConfig() {
  var path = files.join(files.cwd(), "config.json");
  if (!files.exists(path)) return {};
  return JSON.parse(files.read(path));
}

function timestampToken() {
  var d = new Date();
  function pad(n) { return n < 10 ? "0" + n : String(n); }
  return d.getFullYear() + pad(d.getMonth() + 1) + pad(d.getDate()) + "_" +
    pad(d.getHours()) + pad(d.getMinutes()) + pad(d.getSeconds());
}

function formatResult(result) {
  var s = result.snapshot;
  var lines = [
    "========== THS MOBILE PREFLIGHT ==========",
    "status=" + (result.ok ? "PASS" : "FAIL"),
    "package=" + s.package_name,
    "activity=" + s.activity_name,
    "version_name=" + s.app_version_name,
    "version_code=" + s.app_version_code,
    "display=" + s.width + "x" + s.height + " density_dpi=" + s.density_dpi,
    "orientation=" + s.orientation,
    "---------- required accessibility nodes ----------"
  ];
  s.required_ids.forEach(function (item) {
    lines.push((item.present ? "[OK] " : "[MISSING] ") + item.id);
  });
  lines.push("---------- blockers ----------");
  if (s.blockers.length) s.blockers.forEach(function (x) { lines.push("[BLOCK] " + x); });
  else lines.push("[OK] none detected");
  if (result.warnings.length) {
    lines.push("---------- warnings ----------");
    result.warnings.forEach(function (x) { lines.push("[WARN] " + x); });
  }
  if (result.errors.length) {
    lines.push("---------- errors ----------");
    result.errors.forEach(function (x) { lines.push("[FAIL] " + x); });
  }
  lines.push("========== END THS MOBILE PREFLIGHT ==========");
  return lines.join("\n");
}

try {
  var config = loadConfig();
  var result = guard.check(config, { include_metadata: true });
  var text = formatResult(result);
  var outputPath = files.join(files.cwd(), "ths_mobile_preflight_" + timestampToken() + ".txt");
  files.write(outputPath, text);
  console.show();
  console.log(text);
  if (result.ok) {
    dialogs.alert("同花顺手机端预检通过", "关键 UI 节点和运行环境检查通过。\n\n日志: " + outputPath);
  } else {
    dialogs.alert("同花顺手机端预检失败", result.errors.join("\n") + "\n\n日志: " + outputPath);
  }
} catch (e) {
  console.error(e && e.stack ? e.stack : String(e));
  dialogs.alert("同花顺手机端预检异常", String(e));
}
