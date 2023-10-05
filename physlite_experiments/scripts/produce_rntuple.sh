#!/usr/bin/env bash

# Software setup for this script:
# -------------------------------
# export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
# alias setupATLAS=source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
# setupATLAS -c centos7

# Ran on October 5, 2023
# using nightly build with root 6.29 in the externals using
# asetup Athena,main--dev3LCG,2023-09-13T1230
#
# also see Marcin's talk (internal)
# https://indico.cern.ch/event/1324751/contributions/5581789/attachments/2715522/4716448/RNTuple%20in%20Athena%20-%20Sept%202023%20Status.pdf

# testfile, 2023-10-05 pointing to
# /cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/.art/37ef95202635d7967c102fc067115921140ab8b7.art
inputfile=/cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/CampaignInputs/mc21/AOD/mc21_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.recon.AOD.e8453_s3873_r13829/1000events.AOD.29787656._000153.pool.root.1

Derivation_tf.py --CA True --formats PHYSLITE \
    --preExec 'ConfigFlags.Output.StorageTechnology="ROOTRNTUPLE";ConfigFlags.Exec.OutputLevel=4;' \
    --inputAODFile "$inputfile" --outputDAODFile rntuple.root \
    --maxEvents 50
