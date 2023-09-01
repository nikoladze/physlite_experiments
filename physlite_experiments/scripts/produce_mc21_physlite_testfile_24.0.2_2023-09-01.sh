#!/usr/bin/env sh

# adapted from https://gitlab.cern.ch/atlas/athena/-/blob/release/24.0.2/PhysicsAnalysis/DerivationFramework/DerivationFrameworkART/DerivationFrameworkPHYS/test/test_mc21PHYSLITE.sh
# permalink https://gitlab.cern.ch/atlas/athena/-/blob/23eb61af3a0e85ee7c6d6fce532fc4e998575d95/PhysicsAnalysis/DerivationFramework/DerivationFrameworkART/DerivationFrameworkPHYS/test/test_mc21PHYSLITE.sh

# Software setup for this script:
# -------------------------------
# export ATLAS_LOCAL_ROOT_BASE=/cvmfs/atlas.cern.ch/repo/ATLASLocalRootBase
# alias setupATLAS=source ${ATLAS_LOCAL_ROOT_BASE}/user/atlasLocalSetup.sh
# setupATLAS -c centos7
# asetup Athena,24.0.2

# referenced test AOD file on cvmfs was available on 2023-09-01, pointing to
# /cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/.art/37ef95202635d7967c102fc067115921140ab8b7.art

Derivation_tf.py \
    --CA True \
    --inputAODFile /cvmfs/atlas-nightlies.cern.ch/repo/data/data-art/CampaignInputs/mc21/AOD/mc21_13p6TeV.601229.PhPy8EG_A14_ttbar_hdamp258p75_SingleLep.recon.AOD.e8453_s3873_r13829/1000events.AOD.29787656._000153.pool.root.1 \
    --outputDAODFile art.pool.root \
    --formats PHYSLITE \
    --maxEvents 50 \
