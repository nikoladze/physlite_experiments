import awkward as ak
import numpy as np
from coffea.nanoevents.methods import vector

ak.behavior.update(vector.behavior)


class xAODEvents:
    @property
    def _events(self):
        return self.behavior["__events__"][0]

    def element_link(self, links):
        links = links[links.m_persKey != 0]
        collection1 = self
        keys = np.unique(ak.to_numpy(ak.flatten(links.m_persKey, axis=None)))
        if len(keys) >= 2:
            collections = [
                self._events[self._events.branch_names[key]]
                for key in keys
            ]
            return collections, keys, links.m_persIndex, links.m_persKey
            # TODO: how to handle thinned tracks - maybe the ones with m_persKey == 0?
            # need to filter, but then maybe the structure changes (becomes IndexedArray)
            # but maybe that is not really a problem, because we use it flattened below?

            # TODO: now need to construct a union as content
        else:
            self._events.branch_names # TODO: seems i need to touch this first - what's going on?
            collection2 = self._events[
                self._events.branch_names[keys[0]]
            ]
            (
                offsets_collection2,
                content_collection2,
            ) = collection2.layout.offsets_and_flatten()
        offsets_outer, _ = collection1.layout.offsets_and_flatten()
        _, content_links = links.layout.offsets_and_flatten()
        offsets_inner, _ = content_links.offsets_and_flatten()
        # TODO: probably this doesn't work anymore if the array became indexed/masked?
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
    def trackParticles(self):
        return self.element_link(self.trackParticleLinks)

    @property
    def trackParticle(self):
        return self.trackParticles[:, :, 0]


@ak.mixin_class(ak.behavior)
class xAODMuon(xAODParticle):
    @property
    def trackParticle(self):
        # TODO: doesn't really work like that if there are indeed invalid links ...
        # (will have different offsets as muon container)
        # need to deal with the other types of links
        return self._events.CombinedMuonTrackParticles[
            self["combinedTrackParticleLink.m_persIndex"][
                self["combinedTrackParticleLink.m_persKey"] != 0
            ]
        ]


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
