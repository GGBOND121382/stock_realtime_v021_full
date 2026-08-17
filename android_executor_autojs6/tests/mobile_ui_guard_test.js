"use strict";

var assert = require("assert");
var guard = require("../lib/mobile_ui_guard.js");

function baseSnapshot() {
  return {
    package_name: "com.hexin.plat.android",
    app_version_name: "11.55.03",
    width: 1080,
    height: 2400,
    orientation: "portrait",
    required_ids: guard.DEFAULT_REQUIRED_IDS.map(function (id) { return { id: id, present: true }; }),
    blockers: []
  };
}

var ok = guard.evaluateSnapshot(baseSnapshot(), {
  ths_package: "com.hexin.plat.android",
  expected_orientation: "portrait",
  expected_ths_version_name: "11.55.03",
  ui_timeout_ms: 5000,
  field_verify_timeout_ms: 700
});
assert.strictEqual(ok.ok, true);

var missing = baseSnapshot();
missing.required_ids[2].present = false;
assert.strictEqual(guard.evaluateSnapshot(missing, {}).ok, false);

var blocked = baseSnapshot();
blocked.blockers = ["验证码"];
assert.strictEqual(guard.evaluateSnapshot(blocked, {}).ok, false);

var wrongPackage = baseSnapshot();
wrongPackage.package_name = "com.example.other";
assert.strictEqual(guard.evaluateSnapshot(wrongPackage, {}).ok, false);

var wrongVersion = baseSnapshot();
assert.strictEqual(guard.evaluateSnapshot(wrongVersion, { expected_ths_version_name: "99.0" }).ok, false);

var wrongOrientation = baseSnapshot();
assert.strictEqual(guard.evaluateSnapshot(wrongOrientation, { expected_orientation: "landscape" }).ok, false);

var warning = guard.evaluateSnapshot(baseSnapshot(), { ui_timeout_ms: 200, field_verify_timeout_ms: 100 });
assert.strictEqual(warning.ok, true);
assert.ok(warning.warnings.length >= 2);

console.log("mobile_ui_guard tests: OK");
