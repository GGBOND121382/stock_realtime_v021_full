"use strict";

function cleanText(value) {
  return String(value === null || value === undefined ? "" : value)
    .replace(/\u00a0/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function cleanLabel(value) {
  return cleanText(value).replace(/[：:]/g, "").replace(/\s+/g, "");
}

function numeric(value) {
  var text = cleanText(value).replace(/,/g, "");
  var n = Number(text);
  return isFinite(n) ? n : null;
}

function numericEqual(a, b) {
  var x = numeric(a);
  var y = numeric(b);
  return x !== null && y !== null && Math.abs(x - y) < 0.000001;
}

function findLabeledValue(texts, labels) {
  var normalizedLabels = labels.map(cleanLabel);
  for (var i = 0; i < texts.length; i++) {
    var current = cleanText(texts[i]);
    var currentLabel = cleanLabel(current);
    for (var j = 0; j < normalizedLabels.length; j++) {
      var label = normalizedLabels[j];
      if (currentLabel === label && i + 1 < texts.length) {
        return cleanText(texts[i + 1]);
      }

      var rawLabel = cleanText(labels[j]);
      var inline = new RegExp("^" + rawLabel.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + "[：:\\s]+(.+)$");
      var match = current.match(inline);
      if (match) return cleanText(match[1]);
    }
  }
  return "";
}

function parseConfirmationTexts(values) {
  var texts = (values || []).map(cleanText).filter(function (x) { return x.length > 0; });
  var title = "";
  for (var i = 0; i < texts.length; i++) {
    if (texts[i].indexOf("委托买入确认") >= 0 || texts[i].indexOf("委托卖出确认") >= 0) {
      title = texts[i];
      break;
    }
  }
  return {
    title: title,
    account: findLabeledValue(texts, ["账户", "股东账户"]),
    name: findLabeledValue(texts, ["名称", "证券名称"]),
    code: findLabeledValue(texts, ["代码", "证券代码"]),
    qty: findLabeledValue(texts, ["数量", "委托数量"]),
    price: findLabeledValue(texts, ["价格", "委托价格"]),
    texts: texts
  };
}

function validateConfirmation(order, parsed) {
  parsed = parsed || {};
  var errors = [];
  var side = String(order && order.side || "").toLowerCase();
  if (side !== "buy" && side !== "sell") {
    errors.push("invalid expected side=" + String(order && order.side));
    return { ok: false, errors: errors };
  }

  var expectedTitle = side === "sell" ? "委托卖出确认" : "委托买入确认";
  if (String(parsed.title || "").indexOf(expectedTitle) < 0) {
    errors.push("title expected=" + expectedTitle + " actual=" + String(parsed.title || "<missing>"));
  }
  if (!parsed.code || String(parsed.code).replace(/\D/g, "") !== String(order.code)) {
    errors.push("code expected=" + order.code + " actual=" + String(parsed.code || "<missing>"));
  }
  if (!parsed.qty || !numericEqual(parsed.qty, order.qty)) {
    errors.push("qty expected=" + order.qty + " actual=" + String(parsed.qty || "<missing>"));
  }
  if (!parsed.price || !numericEqual(parsed.price, Number(order.submit_price).toFixed(2))) {
    errors.push("price expected=" + Number(order.submit_price).toFixed(2) + " actual=" + String(parsed.price || "<missing>"));
  }
  return { ok: errors.length === 0, errors: errors };
}

var REJECT_PATTERNS = [
  /余额不足/,
  /资金不足/,
  /可用资金不足/,
  /不允许卖空/,
  /价格不合法/,
  /委托价格不合法/,
  /数量必须/,
  /委托数量必须/,
  /股票代码不存在/,
  /证券代码不存在/,
  /停牌/,
  /废单/,
  /超出.*涨跌停/,
  /超过.*涨停/,
  /低于.*跌停/,
  /权限不足/,
  /禁止交易/,
  /不支持.*交易/
];

var NEGATED_SUBMIT_PATTERNS = [
  /委托未提交/,
  /委托提交失败/,
  /提交委托失败/,
  /委托已提交失败/,
  /委托未成功/,
  /提交未成功/,
  /未能提交/
];

function classifyResultMessage(message) {
  var text = cleanText(message);
  if (!text) return { outcome: "unknown", message: text };

  // Never let the positive substring "委托已提交" override an explicit failure/
  // negation in the same message. Such wording is treated as unknown so the caller
  // cannot convert an ambiguous broker result into a submitted terminal state.
  for (var n = 0; n < NEGATED_SUBMIT_PATTERNS.length; n++) {
    if (NEGATED_SUBMIT_PATTERNS[n].test(text)) return { outcome: "unknown", message: text };
  }

  if (/委托已提交/.test(text)) return { outcome: "submitted", message: text };
  for (var i = 0; i < REJECT_PATTERNS.length; i++) {
    if (REJECT_PATTERNS[i].test(text)) return { outcome: "rejected", message: text };
  }
  return { outcome: "unknown", message: text };
}

module.exports = {
  cleanText: cleanText,
  numericEqual: numericEqual,
  parseConfirmationTexts: parseConfirmationTexts,
  validateConfirmation: validateConfirmation,
  classifyResultMessage: classifyResultMessage
};
