##!/bin/bash
groupadd havenask
useradd -l -u 0 -G havenask -md /home/root -s /bin/bash root
echo -e "\nroot ALL=(ALL) NOPASSWD:ALL\n" >> /etc/sudoers
echo "PS1='[\u@havenask \w]\$'" > /etc/profile
echo "export TERM='xterm-256color'" >> /etc/profile
echo "/opt/conda310/lib" >> /etc/ld.so.conf.d/python310-libs.conf
