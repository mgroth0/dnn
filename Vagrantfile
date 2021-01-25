# -*- mode: ruby -*-
# vi: set ft=ruby :
Vagrant.configure("2") do |config|

  config.vm.box = "singularityware/singularity-2.4"

  # hanging on  "Creating SIF file..."
  # config.vm.box = "sylabs/singularity-3.2-ubuntu-bionic64"

  config.vm.synced_folder ".", "/home/dnn"

  # config.disksize.size = '10GB' # plugin not installed in OM, also not a big deal any more

  # config.vm.provider "virtualbox" do |vb|
  #   vb.memory = "10240" # Assign 10 GB RAM (default 1G). Will be sufficient for building Singularity image.
  # end

end
