import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)


class xAODEvents:
    @property
    def _events(self):
        return self.behavior["__events__"][0]

    def element_link(self, links):
        collection1 = self
        keys = np.unique(ak.to_numpy(ak.flatten(links.m_persKey, axis=None)))
        if len(keys) != 1:
            raise NotImplementedError(
                f"Can only link into 1 other collection (got references to {len(keys)})"
            )
        collection2 = self._events[
            self._events.layout.parameters["branch_names"][str(keys[0])]
        ]
        (
            offsets_collection2,
            content_collection2,
        ) = collection2.layout.offsets_and_flatten()
        offsets_outer, _ = collection1.layout.offsets_and_flatten()
        _, content_links = links.layout.offsets_and_flatten()
        offsets_inner, _ = content_links.offsets_and_flatten()
        return ak.Array(
            ak.layout.ListOffsetArray64(
                offsets_outer,
                ak.layout.ListOffsetArray64(
                    offsets_inner,
                    ak.layout.IndexedArray64(
                        ak.layout.Index64(
                            ak.flatten(
                                links.m_persIndex + np.array(offsets_collection2[:-1]),
                                axis=None,
                            )
                        ),
                        content_collection2,
                    ),
                ),
            )
        )


@ak.mixin_class(ak.behavior)
class xAODParticle(vector.PtEtaPhiELorentzVector, xAODEvents):
    @property
    def mass(self):
        return self.m


@ak.mixin_class(ak.behavior)
class xAODElectron(xAODParticle):
    @property
    def trackParticle(self):
        return self.element_link(self.trackParticleLinks)


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
