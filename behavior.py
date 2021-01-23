import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)


class xAODEvents:
    @property
    def _events(self):
        return self.behavior["__events__"][0]


@ak.mixin_class(ak.behavior)
class xAODParticle(vector.PtEtaPhiELorentzVector, xAODEvents):
    @property
    def mass(self):
        return self.m


@ak.mixin_class(ak.behavior)
class xAODTrackParticle(vector.LorentzVector, xAODEvents):
    "see https://gitlab.cern.ch/atlas/athena/-/blob/21.2/Event/xAOD/xAODTracking/Root/TrackParticle_v1.cxx#L82"

    @property
    def theta(self):
        return self["theta"]

    @property
    def phi(self):
        return self["phi"]

    @property
    def p(self):
        return 1.0 / np.abs(self.qOverP)

    @property
    def x(self):
        return self.p * np.sin(self.theta) * np.cos(self.phi)

    @property
    def y(self):
        return self.p * np.sin(self.theta) * np.sin(self.phi)

    @property
    def z(self):
        return self.p * np.cos(self.theta)

    @property
    def t(self):
        return np.sqrt(139.570 ** 2 + self.x ** 2 + self.y ** 2 + self.z ** 2)
