"use strict";

var DEFAULT_REQUIRED_IDS = [
  "com.hexin.plat.android:id/auto_stockcode",
  "com.hexin.plat.android:id/stockvolume",
  "com.hexin.plat.android:id/stockprice",
  "com.hexin.plat.android:id/btn_transaction"
];

var DEFAULT_BLOCKERS = [
  "验证码",
  "重新登录",
  "登录超时",
  "风险提示",
  "系统维护",
  "网络异常",
  "人脸",
  "指纹",
  "安全键盘",
  "委托买入确认",
  "委托卖出确认",
  "委托已提交"
];

function safeValue(fn, fallback) {
  try {
    var value = fn();
    return value === null || value === undefined ? fallback : value;
  } catch (e) {
    return fallback;
  }
}

function regexEscape(text) {
  return String(text).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function normalizeOrientation(value) {
  var v = String(value || "").toLowerCase();
  if (v === "portrait" || v === "竖屏") return "portrait";
  if (v === "landscape" || v === "横屏") return "landscape";
  return "";
}

function evaluateSnapshot(snapshot, config) {
  config = config || {};
  snapshot = snapshot || {};
  var errors = [];
  var warnings = [];
  var expectedPackage = String(config.ths_package || "com.hexin.plat.android");

  if (String(snapshot.package_name || "") !== expectedPackage) {
    errors.push("current package is not THS: " + String(snapshot.package_name || "<empty>"));
  }

  var expectedOrientation = normalizeOrientation(config.expected_orientation);
  if (expectedOrientation && String(snapshot.orientation || "") !== expectedOrientation) {
    errors.push("orientation mismatch: expected=" + expectedOrientation + " actual=" + String(snapshot.orientation || "unknown"));
  }

  if (config.expected_ths_version_name) {
    if (!snapshot.app_version_name) {
      errors.push("unable to read THS version while expected_ths_version_name is pinned");
    } else if (String(config.expected_ths_version_name) !== String(snapshot.app_version_name)) {
      errors.push("THS version mismatch: expected=" + config.expected_ths_version_name + " actual=" + snapshot.app_version_name);
    }
  }

  var required = Array.isArray(snapshot.required_ids) ? snapshot.required_ids : [];
  for (var i = 0; i < required.length; i++) {
    if (!required[i].present) errors.push("required accessibility node missing: " + required[i].id);
  }

  var blockers = Array.isArray(snapshot.blockers) ? snapshot.blockers : [];
  if (blockers.length) {
    errors.push("blocking THS UI detected: " + blockers.join(" | "));
  }

  var uiTimeout = Number(config.ui_timeout_ms || 5000);
  var verifyTimeout = Number(config.field_verify_timeout_ms || 700);
  if (!isFinite(uiTimeout) || uiTimeout < 1000) warnings.push("ui_timeout_ms below 1000ms may cause false missing-node failures");
  if (!isFinite(verifyTimeout) || verifyTimeout < 300) warnings.push("field_verify_timeout_ms below 300ms may cause false read-back failures");

  if (snapshot.width && snapshot.height && snapshot.width === snapshot.height) {
    warnings.push("square display metrics are unusual; verify split-screen/floating-window state");
  }

  return { ok: errors.length === 0, errors: errors, warnings: warnings };
}

function compareBaseline(baseline, snapshot) {
  baseline = baseline || {};
  snapshot = snapshot || {};
  var errors = [];
  if (baseline.package_name && snapshot.package_name && baseline.package_name !== snapshot.package_name) {
    errors.push("package changed during batch: " + baseline.package_name + " -> " + snapshot.package_name);
  }
  if (baseline.orientation && snapshot.orientation && baseline.orientation !== snapshot.orientation) {
    errors.push("orientation changed during batch: " + baseline.orientation + " -> " + snapshot.orientation);
  }
  if (baseline.width && baseline.height && snapshot.width && snapshot.height &&
      (Number(baseline.width) !== Number(snapshot.width) || Number(baseline.height) !== Number(snapshot.height))) {
    errors.push("display metrics changed during batch: " + baseline.width + "x" + baseline.height +
      " -> " + snapshot.width + "x" + snapshot.height);
  }
  if (baseline.app_version_name && snapshot.app_version_name &&
      String(baseline.app_version_name) !== String(snapshot.app_version_name)) {
    errors.push("THS version changed during batch: " + baseline.app_version_name + " -> " + snapshot.app_version_name);
  }
  return { ok: errors.length === 0, errors: errors };
}

function getPackageVersion(packageName) {
  return safeValue(function () {
    var pm = context.getPackageManager();
    var info = pm.getPackageInfo(String(packageName), 0);
    return {
      name: String(info.versionName || ""),
      code: String(android.os.Build.VERSION.SDK_INT >= 28 ? info.getLongVersionCode() : info.versionCode)
    };
  }, { name: "", code: "" });
}

function currentOrientation() {
  return safeValue(function () {
    var orientation = context.getResources().getConfiguration().orientation;
    if (orientation === android.content.res.Configuration.ORIENTATION_LANDSCAPE) return "landscape";
    if (orientation === android.content.res.Configuration.ORIENTATION_PORTRAIT) return "portrait";
    return "unknown";
  }, "unknown");
}

function displayMetrics() {
  return safeValue(function () {
    var dm = context.getResources().getDisplayMetrics();
    return { width: Number(dm.widthPixels), height: Number(dm.heightPixels), density_dpi: Number(dm.densityDpi) };
  }, { width: 0, height: 0, density_dpi: 0 });
}

function buildBlockerRegex(blockers) {
  var items = (blockers || []).map(regexEscape).filter(function (x) { return x.length > 0; });
  if (!items.length) return null;
  return new RegExp(".*(?:" + items.join("|") + ").*");
}

function waitForTargetPackage(config, timeoutMs, launchIfNeeded) {
  config = config || {};
  var expected = String(config.ths_package || "com.hexin.plat.android");
  var total = Number(timeoutMs || config.mobile_return_timeout_ms || 3500);
  if (!isFinite(total) || total < 500) total = 3500;

  var start = Date.now();
  var passiveUntil = start + Math.min(1200, total);
  while (Date.now() <= passiveUntil) {
    if (safeValue(function () { return currentPackage(); }, "") === expected) return true;
    sleep(80);
  }

  if (launchIfNeeded !== false) {
    safeValue(function () { app.launchPackage(expected); return true; }, false);
  }

  var deadline = start + total;
  while (Date.now() <= deadline) {
    if (safeValue(function () { return currentPackage(); }, "") === expected) return true;
    sleep(100);
  }

  var err = new Error("THS package did not become foreground: expected=" + expected);
  err.stage = "mobile_target_package_unavailable";
  err.ambiguous = false;
  err.fatal_ui_state = true;
  throw err;
}

function captureSnapshot(config) {
  config = config || {};
  var expectedPackage = String(config.ths_package || "com.hexin.plat.android");
  var requiredIds = Array.isArray(config.mobile_preflight_required_ids) && config.mobile_preflight_required_ids.length ?
    config.mobile_preflight_required_ids : DEFAULT_REQUIRED_IDS;
  var blockerTerms = Array.isArray(config.mobile_preflight_blockers) ? config.mobile_preflight_blockers : DEFAULT_BLOCKERS;
  var pkg = safeValue(function () { return currentPackage(); }, "");
  var required = [];

  for (var i = 0; i < requiredIds.length; i++) {
    var rid = String(requiredIds[i]);
    var present = safeValue(function () { return !!id(rid).findOnce(); }, false);
    required.push({ id: rid, present: present });
  }

  var blockers = [];
  var blockerRegex = buildBlockerRegex(blockerTerms);
  if (blockerRegex) {
    var blockerNode = safeValue(function () { return textMatches(blockerRegex).findOnce(); }, null);
    if (blockerNode) {
      var blockerText = safeValue(function () { return String(blockerNode.text() || blockerNode.desc() || ""); }, "");
      blockers.push(blockerText || "matched configured blocker text");
    }
  }

  var metrics = displayMetrics();
  // Always read version metadata. The previous "lightweight" path returned an empty
  // version and made a pinned expected_ths_version_name fail on every per-order guard.
  var version = getPackageVersion(expectedPackage);
  return {
    captured_at: new Date().toISOString(),
    package_name: pkg,
    activity_name: safeValue(function () { return currentActivity(); }, ""),
    app_version_name: version.name,
    app_version_code: version.code,
    width: metrics.width,
    height: metrics.height,
    density_dpi: metrics.density_dpi,
    orientation: currentOrientation(),
    required_ids: required,
    blockers: blockers
  };
}

function check(config) {
  var snapshot = captureSnapshot(config);
  var evaluation = evaluateSnapshot(snapshot, config);
  return { ok: evaluation.ok, errors: evaluation.errors, warnings: evaluation.warnings, snapshot: snapshot };
}

function assertReady(config) {
  var result = check(config);
  if (result.ok) return result;
  var err = new Error("THS mobile preflight failed: " + result.errors.join("; "));
  err.stage = "mobile_preflight_failed";
  err.ambiguous = false;
  err.fatal_ui_state = true;
  err.preflight = result;
  throw err;
}

module.exports = {
  DEFAULT_REQUIRED_IDS: DEFAULT_REQUIRED_IDS,
  DEFAULT_BLOCKERS: DEFAULT_BLOCKERS,
  normalizeOrientation: normalizeOrientation,
  evaluateSnapshot: evaluateSnapshot,
  compareBaseline: compareBaseline,
  waitForTargetPackage: waitForTargetPackage,
  captureSnapshot: captureSnapshot,
  check: check,
  assertReady: assertReady
};