"use strict";

function joinUrl(base, path) {
  return String(base).replace(/\/+$/, "") + "/" + String(path).replace(/^\/+/, "");
}

function fetchLatest(config, experiment) {
  var path = "api/v1/execution/latest";
  if (experiment !== undefined && experiment !== null && String(experiment).trim()) {
    path += "?experiment=" + encodeURIComponent(String(experiment).trim());
  }
  var url = joinUrl(config.api_base_url, path);
  var headers = { "Accept": "application/json" };
  if (config.api_token) headers["Authorization"] = "Bearer " + String(config.api_token);
  var response = http.get(url, {
    headers: headers,
    timeout: Number(config.request_timeout_ms || 5000)
  });
  if (!response) throw new Error("empty HTTP response");
  if (Number(response.statusCode) === 204) return null;
  if (Number(response.statusCode) !== 200) {
    throw new Error("HTTP " + response.statusCode + ": " + response.body.string());
  }
  return JSON.parse(response.body.string());
}

module.exports = { fetchLatest: fetchLatest };
