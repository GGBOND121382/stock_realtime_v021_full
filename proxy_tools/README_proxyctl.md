# VS Code Remote-SSH 代理开关脚本

这个目录包含一个脚本：

```text
proxyctl.sh
```

用途：在远程服务器上一键开启/关闭代理配置，让服务器上的命令行程序、Git、数据下载脚本走 VS Code / SSH 反向端口转发出来的本地 Clash 代理。

默认代理地址：

```text
http://127.0.0.1:17890
```

它对应你的 SSH 配置：

```sshconfig
RemoteForward 17890 127.0.0.1:7890
```

也就是：

```text
远程服务器 127.0.0.1:17890
        ↓ SSH 反向隧道
本地电脑 127.0.0.1:7890 Clash
        ↓
外网 / GitHub / 数据源
```

---

## 1. 上传到服务器

把 `proxyctl.sh` 上传到远程服务器，例如放到家目录：

```bash
scp proxyctl.sh root@服务器IP:~/
```

或者你已经在 VS Code Remote-SSH 里，可以直接拖到服务器目录。

然后在服务器上执行：

```bash
chmod +x ./proxyctl.sh
```

---

## 2. 开启代理

在远程服务器终端执行：

```bash
source ./proxyctl.sh on
```

这一步会做三件事：

1. 设置当前终端的代理环境变量；
2. 写入 `~/.proxy_env`，方便之后手动 `source ~/.proxy_env`；
3. 设置 Git 全局代理：

```bash
git config --global http.proxy http://127.0.0.1:17890
git config --global https.proxy http://127.0.0.1:17890
git config --global http.version HTTP/1.1
```

开启后可以直接运行：

```bash
git pull origin master
python3 your_download_script.py
curl -I https://github.com
```

---

## 3. 关闭代理

在远程服务器终端执行：

```bash
source ./proxyctl.sh off
```

这一步会：

1. 清理当前终端的代理环境变量；
2. 删除 `~/.proxy_env`；
3. 清理 Git 的全局代理配置：

```bash
git config --global --unset http.proxy
git config --global --unset https.proxy
```

说明：脚本默认保留 `git config --global http.version HTTP/1.1`，因为它通常对代理访问 GitHub 更稳定。如果你也想删除它，可以手动执行：

```bash
git config --global --unset http.version
```

---

## 4. 查看代理状态

执行：

```bash
./proxyctl.sh status
```

它会检查：

1. 当前环境变量里有没有代理；
2. `~/.proxy_env` 是否存在；
3. Git 全局代理配置；
4. 服务器的 `17890` 端口是否由 `sshd` 监听；
5. 通过代理访问 GitHub 是否成功。

正常时，端口检查可能看到类似：

```text
LISTEN ... 127.0.0.1:17890 ... users:(("sshd",pid=1298,...))
```

这表示正常。`17890` 本来就应该由服务器上的 `sshd` 进程监听。

---

## 5. 不改变当前 shell，只给某个命令临时加代理

可以用 `run` 子命令：

```bash
./proxyctl.sh run git pull origin master
```

或者：

```bash
./proxyctl.sh run python3 your_download_script.py
```

这个模式不会长期修改当前 shell 环境变量，只会让这一条命令走代理。

---

## 6. 修改端口

默认端口是：

```text
17890
```

如果你以后把 VS Code / SSH 的反向端口改成了 `17891`，可以这样临时指定：

```bash
PROXY_PORT=17891 source ./proxyctl.sh on
```

查看状态：

```bash
PROXY_PORT=17891 ./proxyctl.sh status
```

---

## 7. 重要限制

这个脚本**不会创建 SSH 隧道**。

它只是在服务器上配置代理环境变量，让程序访问：

```text
127.0.0.1:17890
```

所以必须同时满足：

1. VS Code Remote-SSH 或独立 SSH 连接还在；
2. SSH 配置里有：

```sshconfig
RemoteForward 17890 127.0.0.1:7890
```

3. 本地 Clash 正在运行；
4. 本地 Clash 的端口确实是 `7890`。

如果 VS Code / SSH 断开，服务器上的 `127.0.0.1:17890` 隧道会消失。此时即使执行过：

```bash
source ./proxyctl.sh on
```

服务器后台服务也无法继续通过你本地电脑联网。

---

## 8. 推荐用法

### 临时手动操作

```bash
source ./proxyctl.sh on
git pull origin master
source ./proxyctl.sh off
```

### 单条命令临时走代理

```bash
./proxyctl.sh run git pull origin master
```

### tmux 里跑下载脚本

```bash
tmux
source ./proxyctl.sh on
python3 your_download_script.py
```

注意：如果你关闭本地电脑、断开 VS Code、断开 SSH，tmux 里的脚本仍然会失去这个代理隧道。

---

## 9. 如果后台服务必须长期联网

如果你的服务器服务需要在你断开 VS Code 后继续联网下载数据，不建议依赖这个 SSH 反向隧道。

更稳的方式是在服务器本机部署代理，例如服务器自己运行 mihomo / clash / v2ray，然后让服务直接访问服务器本机代理端口。
