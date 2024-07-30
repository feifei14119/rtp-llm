##!/bin/bash
groupadd havenask
useradd -l -u 1000 -G havenask -md /home/feifei -s /bin/bash feifei
echo -e "\nfeifei ALL=(ALL) NOPASSWD:ALL\n" >> /etc/sudoers
echo "PS1='[\u@havenask \w]\$'" > /etc/profile
echo "export TERM='xterm-256color'" >> /etc/profile
mkdir -p /mnt/ram
mount -t ramfs -o size=20g ramfs /mnt/ram
chmod a+rw /mnt/ram
