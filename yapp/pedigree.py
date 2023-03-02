# -*- coding: utf-8 -*-
"""Module pedigree.py from yapp

This module exports classes and functions to read and manipulate
pedigrees.  A pedigree is a directed graph of familial relationships
(father, mother, offspring).

"""
import sys
import warnings
import logging
from . import MALE, FEMALE

logger = logging.getLogger(__name__)


class PedNode:
    """A node in the pedigree that consists of an individual and its
    direct relatives.

        Attributes
        ----------
        indiv : obj
           the individual data. Must be hashable.
        father : obj
           the individual's father node. Must be a PedNode object.
        mother : obj
           the individual's mother node. Must be a PedNode object.
        children : list of obj
           the individual's offspring. Must be PedNode objects.

    """

    def __init__(self, indiv, father=None, mother=None, sex=None):
        self.__indiv = indiv
        self.__father = None
        self.__mother = None
        self.__sex = None
        self.__gen = None
        self.father = father
        self.mother = mother
        self.sex = sex
        self.children = []

    def __str__(self):
        output = (
            f"*** {self.__indiv} ***\n"
            f"Father   : {self.father is None and 'NULL' or self.father.indiv}\n"  # noqa
            f"Mother   : {self.mother is None and 'NULL' or self.mother.indiv}\n"  # noqa
            f"Children : [ {*self.children,} ]"
        )
        return output

    @property
    def gen(self):
        if self.__gen is None:
            self.__gen = self.compute_gen()
        return self.__gen

    @gen.setter
    def gen(self, v):
        if v is not None:
            raise ValueError("PedNode.gen can only be assigned None")
        self.__gen = None

    def compute_gen(self):
        if self.father is None and self.mother is None:
            return 0
        else:
            if self.father is None:
                return self.mother.gen + 1
            elif self.mother is None:
                return self.father.gen + 1
            else:
                return max(self.mother.gen + 1, self.father.gen + 1)

    @property
    def indiv(self):
        """Individual data"""
        return self.__indiv

    @indiv.setter
    def indiv(self, v):
        warnings.warn("[PedNode] : indiv attribute is read-only.")

    @property
    def father(self):
        """Father node"""
        return self.__father

    @father.setter
    def father(self, indiv):
        if indiv is None:
            self.__father = None
        elif self.__father is None:
            assert type(indiv) is PedNode
            self.__father = indiv
        else:
            raise ValueError("Setting a new father is not possible.")

    @property
    def mother(self):
        """Mother node"""
        return self.__mother

    @mother.setter
    def mother(self, indiv):
        if indiv is None:
            self.__mother = None
        elif self.__mother is None:
            assert type(indiv) is PedNode
            self.__mother = indiv
        else:
            raise ValueError("Setting a new mother is not possible.")

    @property
    def sex(self):
        """FEMALE or MALE or None"""
        return self.__sex

    @sex.setter
    def sex(self, v):
        if self.sex and v != self.sex:
            raise ValueError(f"Sex mismatch for individual {self.indiv}")
        else:
            assert v == MALE or v == FEMALE or v is None
            self.__sex = v

    def add_child(self, child):
        """Add a child to the node."""
        assert type(child) is PedNode
        if child not in self.children:
            self.children.append(child)


class Pedigree:
    """Pedigree relationships between individuals"""

    def __init__(self, members=[]):
        """
        Parameters
        ----------
        members : list of obj
            List of members to add to the pedigree at initialization.
            The type of objects in the list must be hashable.
        """
        self.nodes = {}
        self.__founders = None
        self.__non_founders = None
        self.__families = None
        for m in members:
            self.add_node(m)

    def __iter__(self):
        """Iterator on nodes in the pedigree.
        Parents are guaranteed to precede offspring.
        """
        for n in sorted(self.nodes.values(), key=lambda x: x.gen):
            yield n

    def to_tuples(self):
        """
        Stores relationships in the pedigree in tuples (indiv,father,mother)
        returns a list of such tuples
        """
        rels = []
        for node in self:
            fa = (node.father is None) and "0" or node.father.indiv
            mo = (node.mother is None) and "0" or node.mother.indiv
            rels.append((node.indiv, fa, mo))
        return rels

    @classmethod
    def from_tuples(cls, rels):
        ped = cls()
        for rel in rels:
            if rel[1] != "0":
                ped.set_father(rel[0], rel[1])
            if rel[2] != "0":
                ped.set_mother(rel[0], rel[2])
            if rel[1] == "0" and rel[2] == "0":
                ped.add_indiv(rel[0])
            _ = ped.nodes[rel[0]]
        return ped

    @classmethod
    def from_fam_file(cls, path, parent_from_FID=False, default_parent=MALE):
        with open(path) as f:
            relations = []
            for ligne in f:
                buf = ligne.split()
                ind = buf[1]
                if buf[2] == "0":
                    fa = None
                    if parent_from_FID and default_parent == MALE:
                        fa = buf[0]
                else:
                    fa = buf[2]
                if buf[3] == "0":
                    mo = None
                    if parent_from_FID and default_parent == FEMALE:
                        mo = buf[0]
                else:
                    mo = buf[3]
                relations.append((ind, fa, mo))
            ped = cls()
            for rel in relations:
                if rel[1] is not None:
                    ped.set_father(rel[0], rel[1])
                if rel[2] is not None:
                    ped.set_mother(rel[0], rel[2])
                if rel[1] is None and rel[2] is None:
                    ped.add_indiv(rel[0])
                _ = ped.nodes[rel[0]]
        return ped

    @classmethod
    def from_pednodes(cls, pednodes):
        """
        Create a new pedigree from a list of pednodes.

        Argument
        --------
        pednodes : list of PedNode objects

        """
        ped = cls()
        for n in pednodes:
            assert type(n) is PedNode
            ped.nodes[n.indiv] = n
        for n in ped.nodes.values():
            if n.father is not None:
                try:
                    _ = ped.nodes[n.father.indiv]
                except KeyError:
                    logger.warning(
                        "When creating new pedigree from pednodes : "
                        "father not found in nodes, deleting"
                    )
                    n.father = None
            if n.mother is not None:
                try:
                    _ = ped.nodes[n.mother.indiv]
                except KeyError:
                    warnings.warn(
                        "When creating new pedigree from pednodes : "
                        "mother not found in nodes, deleting"
                    )
                    n.mother = None
            putative_children = n.children[:]
            for c in putative_children:
                try:
                    _ = ped.nodes[c.indiv]
                except KeyError:
                    warnings.warn(
                        "When creating new pedigree from pednodes : "
                        "child not found, deleting"
                    )
                    n.children.remove(c)
        return ped

    @property
    def founders(self):
        """List of founders in the pedigree (no known parents)"""
        if self.__founders is None:
            self.__founders = [
                x for x in self.nodes.values() if x.father is None and x.mother is None
            ]
        return self.__founders

    @property
    def non_founders(self):
        """List of non founders in the pedigree (at least one know parent)"""
        if self.__non_founders is None:
            self.__non_founders = [
                x
                for x in self.nodes.values()
                if x.father is not None or x.mother is not None
            ]
        return self.__non_founders

    @property
    def families(self):
        """List of Pedigree objects. One for each independant family in the
        dataset"""
        if self.__families is None:
            self.__families = self.build_families()
        return self.__families

    def _reset_families(self):
        """Reset structural information in the pedigree"""
        # reset founders / non_founders / families information
        self.__founders = None
        self.__non_founders = None
        self.__families = None
        for n in self.nodes.values():
            n.gen = None

    def add_indiv(self, indiv):
        """Add a new guy to the pedigree"""
        if indiv in self.nodes:
            logger.debug(f"{indiv} is already there")
        else:
            self.nodes[indiv] = PedNode(indiv)

    def del_indiv(self, indiv):
        """Remove an individual from the pedigree. Links to its node are
        removed.

                Returns
                -------
                the node of the individual if found, else None.

        """
        try:
            n = self.nodes[indiv]
        except KeyError:
            return None
        else:
            del self.nodes[indiv]
            for c in n.children:
                if c.father and c.father == n:
                    c.father = None
                elif c.mother and c.mother == n:
                    c.mother = None
            if n.father:
                n.father.children.remove(n)
            if n.mother:
                n.mother.children.remove(n)
            return n

    def add_node(self, node):
        """Add a node in the pedigree.
        Familial links are restored.

        Argument:
        ---------
        node : PedNode object
            the node to add
        """
        assert type(node) == PedNode
        self.nodes[node.indiv] = node
        if node.father:
            node.father.children.append(node)
        if node.mother:
            node.mother.children.append(node)
        if node.sex:
            if node.sex == MALE:
                for c in node.children:
                    c.father = node
            elif node.sex == FEMALE:
                for c in node.children:
                    c.mother = node

    def set_father(self, offspring, dad):
        """Set that a dad -> offspring relationship

        If called more than once with a different dad a warning is
        raised and the first call takes precedence.  If dad and/or
        offspring are not in the pedigree they will be added.


        Parameters
        ----------
        offspring : object
            offspring to be set. Must be hashable.
        dad : object
           dad to be set. Must be hashable.
        """
        try:
            indiv = self.nodes[offspring]
        except KeyError:
            self.add_indiv(offspring)
            indiv = self.nodes[offspring]

        try:
            father = self.nodes[dad]
            if not father.sex:
                father.sex = MALE
        except KeyError:
            self.add_indiv(dad)
            father = self.nodes[dad]
            father.sex = MALE

        indiv.father = father
        father.add_child(indiv)
        self._reset_families()

    def get_father(self, indiv):
        try:
            member = self.nodes[indiv]
        except KeyError:
            raise KeyError("Individual not found in pedigree")
        return member.father

    def set_mother(self, offspring, mom):
        """Set that a mom -> offspring relationship

        If called more than once with a different mom a warning is
        raised and the first call takes precedence.  If mom and/or
        offspring are not in the pedigree they will be added.

        Parameters
        ----------
        offspring : object
           offspring to be set. Must be hashable.
        mom : object
           mom to be set. Must be hashable.
        """
        try:
            indiv = self.nodes[offspring]
        except KeyError:
            self.add_indiv(offspring)
            indiv = self.nodes[offspring]

        try:
            mother = self.nodes[mom]
            if not mother.sex:
                mother.sex = FEMALE
        except KeyError:
            self.add_indiv(mom)
            mother = self.nodes[mom]
            mother.sex = FEMALE

        indiv.mother = mother
        mother.add_child(indiv)
        self._reset_families()

    def get_mother(self, indiv):
        try:
            member = self.nodes[indiv]
        except KeyError:
            raise KeyError("Individual not found in pedigree")
        return member.mother

    def _get_relatives(self, node, connected=None):
        """Get all individuals connected to a node in the pedigree.

        Parameters
        ----------
        node : obj of class PedNode
           the investigated node
        connected : list of obj of class PedNode
           the list of ancestors and descendants of node, populated by
           calling this method.

        Returns
        -------
        list of obj
           list of PedNode objects, including node

        """
        if connected is None:
            connected = []
        if node in connected:
            return
        connected.append(node)
        for off in node.children:
            self._get_relatives(off, connected)
        if node.father:
            self._get_relatives(node.father, connected)
        if node.mother:
            self._get_relatives(node.mother, connected)
        return connected

    @property
    def unrelated_individuals(self):
        """List of unrelated individuals with no parents and no child in the
        pedigree"""
        return [x.indiv for x in self.founders if len(x.children) == 0]

    def del_unrelated(self):
        """Remove unrelated individuals from the pedigree"""
        torm = self.unrelated_individuals[:]
        for indiv in torm:
            logger.debug(f"Removing {indiv} as it is not related to anyone")
            self.del_indiv(indiv)

    def build_families(self):
        """
        Cluster individuals by families
        """
        founders2go = self.founders[:]
        offspring2go = self.non_founders[:]
        families = []
        while len(founders2go) > 0:
            relatives = self._get_relatives(founders2go[0])
            fam = Pedigree.from_pednodes(relatives)
            families.append(fam)
            for x in relatives:
                try:
                    founders2go.remove(x)
                except ValueError:
                    offspring2go.remove(x)
        return families

    def get_family(self, indiv):
        """Get subpedigree connected to an individual in the pedigree

        Argmuent
        ----------
        indiv: name of the focal individual

        Returns: object
        --------
        A pedigree object with all individuals connected to the focal
        individual in the pedigree

        """
        try:
            node = self.nodes[indiv]
        except KeyError:
            raise ValueError(f"Focal individual {indiv} not found in pedigree")
        if len(self.nodes) > sys.getrecursionlimit():
            sys.setrecursionlimit(len(self.nodes) + 1)
        relatives = self._get_relatives(node)
        fam = Pedigree.from_pednodes(relatives)
        return fam
