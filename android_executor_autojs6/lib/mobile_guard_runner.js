"use strict";

var guard = require("./mobile_ui_guard.js");

function asFatal(result, stage) {
  var err = new Error("THS mobile guard failed: " + result.errors.join("; "));
  err.stage = stage || "mobile_preflight_failed";
  err.ambiguous = false;
  err.fatal_ui_state = true;
  err.preflight = result;
  return err;
}

function hasHardError(result) {
  return result.errors.some(function (message) {
    return message.indexOf("blocking THS UI detected") === 0 ||
      message.indexOf("THS version mismatch") === 0 ||
      message.indexOf("unable to read THS version") === 0 ||
      message.indexOf("orientation mismatch") === 0;
  });
}

function waitUntilReady(config, timeoutMs) {
  var duration = Number(timeoutMs || 1200);
  if (!isFinite(duration) || duration < 0) duration = 1200;
  var deadline = Date.now() + duration;
  var last = null;

  do {
    last = guard.check(config);
    if (last.ok) return last;
    if (hasHardError(last)) throw asFatal(last, "mobile_preflight_failed");
    if (Date.now() >= deadline) break;
    sleep(80);
  } while (true);

  throw asFatal(last || { errors: ["mobile guard produced no result"] }, "mobile_preflight_timeout");
}

function enterAndWait(config, foregroundTimeoutMs, readyTimeoutMs) {
  guard.waitForTargetPackage(config, foregroundTimeoutMs, true);
  return waitUntilReady(config, readyTimeoutMs);
}

module.exports = {
  waitUntilReady: waitUntilReady,
  enterAndWait: enterAndWait
};