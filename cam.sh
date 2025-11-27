#!/bin/bash

set -e
echo "🚨 Starting full camera stack cleanup + reinstall..."

########################################
# 1. REMOVE ANY BROKEN / DEBIAN PACKAGES
########################################
echo "🧹 Removing old libcamera, picamera2, python libs..."
sudo apt purge -y \
    python3-picamera2 \
    python3-libcamera \
    python3-libcamera-apps \
    libcamera0 \
    libcamera-apps \
    libcamera-tools \
    libcamera-ipa \
    libcamera-* || true

sudo apt autoremove -y
sudo apt clean

########################################
# 2. FIX APT SOURCES (RESET TO PI OS)
########################################

echo "🧾 Resetting /etc/apt/sources.list ..."
sudo bash -c 'cat > /etc/apt/sources.list <<EOF
deb http://deb.debian.org/debian bookworm main contrib non-free-firmware
deb http://deb.debian.org/debian bookworm-updates main contrib non-free-firmware
deb http://security.debian.org/debian-security bookworm-security main contrib non-free-firmware
EOF'

echo "🧾 Resetting /etc/apt/sources.list.d/raspi.list ..."
sudo bash -c 'mkdir -p /etc/apt/sources.list.d &&
cat > /etc/apt/sources.list.d/raspi.list <<EOF
deb http://archive.raspberrypi.com/debian/ bookworm main
EOF'

########################################
# 3. FIX RASPBERRY PI GPG KEY
########################################
echo "🔑 Installing raspberrypi-archive key..."
sudo apt install -y --reinstall raspberrypi-archive-keyring || {
    echo "🔑 Manual key install..."
    curl -fsSL https://archive.raspberrypi.com/debian/raspberrypi.gpg.key \
        | sudo gpg --dearmor -o /usr/share/keyrings/raspberrypi-archive-keyring.gpg
}

########################################
# 4. UPDATE + UPGRADE
########################################
echo "🔄 Updating system..."
sudo apt update
sudo apt full-upgrade -y

########################################
# 5. REINSTALL CAMERA STACK
########################################
echo "📦 Installing libcamera + Picamera2 + apps..."
sudo apt install -y \
    libcamera0 \
    libcamera-apps \
    python3-libcamera \
    python3-picamera2

########################################
# 6. ENABLE CAMERA OVERLAYS
########################################
echo "🧩 Ensuring camera overlay is enabled in config.txt..."
sudo sed -i 's/^#*camera_auto_detect=.*/camera_auto_detect=1/' /boot/firmware/config.txt

########################################
# 7. REBOOT NOTICE
########################################
echo "⚠️ All done! A reboot is required."
echo "➡️ Run: sudo reboot"
echo
echo "After reboot, test camera with:"
echo "libcamera-hello"
