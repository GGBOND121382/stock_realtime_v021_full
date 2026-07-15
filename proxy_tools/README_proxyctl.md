# 服务器代理开关

仓库根目录提供稳定入口：

```bash
source ./proxyctl.sh on
source ./proxyctl.sh off
bash ./proxyctl.sh status
```

实际实现位于：

```text
proxy_tools/proxyctl.sh
```

默认代理地址：

```text
http://127.0.0.1:17890
```

它对应 SSH 反向端口转发，例如：

```sshconfig
RemoteForward 17890 127.0.0.1:7890
```

## 开启代理

```bash
source ./proxyctl.sh on
```

该命令会同时完成：

1. 设置当前 shell 的大小写代理环境变量；
2. 写入 `~/.proxy_env`；
3. 强制更新并验证 Git 全局代理：

```bash
git config --global --replace-all http.proxy http://127.0.0.1:17890
git config --global --replace-all https.proxy http://127.0.0.1:17890
git config --global --replace-all http.version HTTP/1.1
```

成功输出中应包含：

```text
[OK] proxy enabled
     git http.proxy=http://127.0.0.1:17890
     git https.proxy=http://127.0.0.1:17890
```

之后可直接执行：

```bash
git pull
```

## 关闭代理

```bash
source ./proxyctl.sh off
```

它会清除当前 shell 代理变量、删除 `~/.proxy_env`，并执行：

```bash
git config --global --unset-all http.proxy
git config --global --unset-all https.proxy
```

`http.version=HTTP/1.1` 默认保留。

## 状态与连通性检查

```bash
bash ./proxyctl.sh status
```

会检查：

- 当前环境变量；
- Git 全局代理配置；
- `127.0.0.1:17890` 是否有 SSH 隧道监听；
- `curl` 是否能通过显式 HTTP 代理访问 GitHub；
- `git ls-remote` 是否能访问本仓库。

## 临时运行单条命令

```bash
bash ./proxyctl.sh run git pull
bash ./proxyctl.sh run python3 your_download_script.py
```

## 修改端口

```bash
PROXY_PORT=17891 source ./proxyctl.sh on
```

## 注意

该脚本不会创建 SSH 隧道。VS Code Remote-SSH 或独立 SSH 连接必须仍然存在，本地 Clash 也必须监听 SSH 配置中指定的端口。

新版脚本不会在被 `source` 时修改父 shell 的 `set -u` 等选项。
