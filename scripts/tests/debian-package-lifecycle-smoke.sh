#!/bin/sh
set -eu

if [ "$#" -ne 2 ]; then
    echo "usage: $0 OLD.deb NEW.deb" >&2
    exit 2
fi

old_pkg=$(realpath "$1")
new_pkg=$(realpath "$2")
[ -f "$old_pkg" ] && [ -f "$new_pkg" ]
[ "$(dpkg-deb -f "$old_pkg" Package)" = vbuf-ml ]
[ "$(dpkg-deb -f "$new_pkg" Package)" = vbuf-ml ]
old_version=$(dpkg-deb -f "$old_pkg" Version)
new_version=$(dpkg-deb -f "$new_pkg" Version)
dpkg --compare-versions "$old_version" lt "$new_version"

apt-get update
apt-get install --yes --no-install-recommends "$old_pkg"
[ "$(dpkg-query -W -f='${Version}' vbuf-ml)" = "$old_version" ]
/usr/bin/vbuf_compat_server --help >/dev/null

apt-get install --yes --no-install-recommends --reinstall "$old_pkg"
[ "$(dpkg-query -W -f='${Version}' vbuf-ml)" = "$old_version" ]

apt-get install --yes --no-install-recommends "$new_pkg"
[ "$(dpkg-query -W -f='${Version}' vbuf-ml)" = "$new_version" ]
/usr/bin/vbuf_compat_server --help >/dev/null

apt-get remove --yes vbuf-ml
[ ! -e /usr/bin/vbuf_compat_server ]
[ ! -e /usr/lib/vbuf-ml/vbuf_compat_server ]

apt-get install --yes --no-install-recommends "$new_pkg"
[ "$(dpkg-query -W -f='${Version}' vbuf-ml)" = "$new_version" ]
/usr/bin/vbuf_compat_server --help >/dev/null

apt-get purge --yes vbuf-ml
if dpkg-query -W -f='${db:Status-Status}' vbuf-ml 2>/dev/null | grep -q .; then
    echo "package remains registered after purge" >&2
    exit 1
fi
[ ! -e /usr/bin/vbuf_compat_server ]
[ ! -e /usr/lib/vbuf-ml/vbuf_compat_server ]
[ ! -e /usr/lib/vbuf-ml/lib/libvbuf_ml.so ]
[ ! -e /etc/systemd/system/vbuf-ml.service ]

printf 'DEBIAN_PACKAGE_LIFECYCLE_SMOKE=PASS\nold=%s\nnew=%s\n' "$old_version" "$new_version"
