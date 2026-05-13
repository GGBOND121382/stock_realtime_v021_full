可以，README 里直接放这一段。

```bash
# 推荐：快速拉取干净分支，只拉 clean-master，不拉旧 master 历史
git clone --single-branch -b clean-master --depth 1 --filter=blob:none https://github.com/GGBOND121382/stock_realtime_v021_full.git
```

如果 HTTPS 报 `HTTP2 framing layer` 或 `GnuTLS recv error`，先执行：

```bash
git config --global http.version HTTP/1.1
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
```

然后再 clone：

```bash
git clone --single-branch -b clean-master --depth 1 --filter=blob:none https://github.com/GGBOND121382/stock_realtime_v021_full.git
```

如果还想进一步避免 checkout `saved_data` 下的中间产物目录，可以用 sparse 版：

```bash
git clone --single-branch -b clean-master --depth 1 --filter=blob:none --sparse https://github.com/GGBOND121382/stock_realtime_v021_full.git
cd stock_realtime_v021_full

git sparse-checkout init --no-cone

cat > .git/info/sparse-checkout <<'EOF'
/*
!/saved_data/*/00*/**
!/saved_data/*/01*/**
!/saved_data/*/02*/**
!/saved_data/*/03*/**
!/saved_data/*/04*/**
!/saved_data/*/05*/**
!/saved_data/*/06*/**
!/saved_data/*/07*/**
!/saved_data/*/08*/**
!/saved_data/*/09*/**
!/saved_data/*/10*/**
EOF

git sparse-checkout reapply
```

如果 `clean-master` 没有推上去，只能先拉当前 `master` 的浅克隆版本：

```bash
git clone --single-branch -b master --depth 1 --filter=blob:none https://github.com/GGBOND121382/stock_realtime_v021_full.git
```

README 里建议写主命令：

```bash
git clone --single-branch -b clean-master --depth 1 --filter=blob:none https://github.com/GGBOND121382/stock_realtime_v021_full.git
```

这条最短，也最适合告诉别人快速拉取。



# .git清理

清理命令：
git reflog expire --expire=now --all
git gc --prune=now --aggressive

查看命令：
du -sh .git
git count-objects -vH

# 上传命令
git config --global http.version HTTP/1.1
git config --global http.lowSpeedLimit 0
git config --global http.lowSpeedTime 999999
git push --progress origin master:master