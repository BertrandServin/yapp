# -*- coding: utf-8 -*-
"""Module pedigree.py from yapp

This module exports classes and functions to read and manipulate pedigrees.
A pedigree is a directed graph of familial relationships (father, mother, offspring). 
"""
import warnings
from collections import defaultdict

class PedNode():
    """A node in the pedigree that consists of an individual and its direct relatives.

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
    def __init__(self,indiv,father=None,mother=None):
        self.__indiv=indiv
        self.__father=None
        self.__mother=None
        self.father=father
        self.mother=mother
        self.children=[]

    def __str__(self):
        output = (
            f"*** {self.__indiv} ***\n"
            f"Father : {self.father is None and 'NULL' or self.father.indiv}\n"
            f"Mother : {self.mother is None and 'NULL' or self.mother.indiv}\n"
            )
        return output
    
    @property
    def indiv(self):
        """Individual data"""
        return self.__indiv
    @indiv.setter
    def indiv(self,v):
        warnings.warn("[PedNode] : indiv attribute is read-only.")
    
    @property
    def father(self):
        """Father node"""
        return self.__father
    @father.setter
    def father(self, indiv):
        if self.father != None and indiv != self.father:
            raise ValueError('Setting a new father is not possible.')
        if indiv is not None:
            assert type(indiv) is PedNode
            self.__father = indiv

    @property
    def mother(self):
        """Mother node """
        return self.__mother
    @mother.setter
    def mother(self, indiv):
        if self.mother != None and indiv != self.mother:
            raise ValueError('Setting a new mother is not possible.')
        if indiv is not None:
            assert type(indiv) is PedNode
            self.__mother = indiv

    def add_child(self,child):
        """Add a child to the node."""
        assert type(child) is PedNode
        if child not in self.children:
            self.children.append(child)

class Pedigree():
    ''' Pedigree relationships between individuals'''
    def __init__(self,members=[]):
        """
        Parameters
        ----------
        members : list of obj
            List of members to add to the pedigree at initialization.
            The type of objects in the list must be hashable.
        """
        self.nodes={}
        self.__founders = None
        self.__non_founders = None
        self.__families = None
        for m in members:
            self.add_node(m)

    @classmethod
    def from_fam_file(cls, path):
        with open(path) as f:
            relations = []
            for ligne in f:
                buf = ligne.split()
                ind = buf[1]
                if buf[2] == '0':
                    f = None
                else:
                    f = buf[2]
                if buf[3] == '0':
                    m = None
                else:
                    m = buf[3]
                relations.append((ind, f, m))
            ped = cls()
            for rel in relations:
                if rel[1] is not None:
                    ped.set_father(rel[0],rel[1])
                if rel[2] is not None:
                    ped.set_mother(rel[0],rel[2])
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
                    f = ped.nodes[n.father.indiv]
                except KeyError:
                    warnings.warn("When creating new pedigree from pednodes : father not found in nodes, deleting")
                    n.father = None
            if n.mother is not None:
                try:
                    m = ped.nodes[n.mother.indiv]
                except KeyError:
                    warnings.warn("When creating new pedigree from pednodes : mother not found in nodes, deleting")
                    n.mother = None
            putative_children = n.children[:]
            for c in putative_children:
                try:
                    c_data = ped.nodes[c.indiv]
                except KeyError:
                    warnings.warn("When creating new pedigree from pednodes : child not found, deleting")
                    n.children.remove(c)
        return ped
    
    @property
    def founders(self):
        """List of founders in the pedigree (no known parents) """
        if self.__founders is None:
            self.__founders = [x for x in self.nodes.values() if x.father is None and x.mother is None]
        return self.__founders

    @property
    def non_founders(self):
        """List of non founders in the pedigree (at least one know parent) """
        if self.__non_founders is None:
            self.__non_founders = [x for x in self.nodes.values() if x.father is not None or x.mother is not None]
        return self.__non_founders

    @property
    def families(self):
        """List of Pedigree objects. One for each independant family in the dataset """
        if self.__families is None:
            self.__families = self.build_families()
        return self.__families
        
    def _reset_families(self):
        """Reset structural information in the pedigree """
        ## reset founders / non_founders / families information
        self.__founders = None
        self.__non_founders = None
        self.__families = None
        
    def add_node(self,indiv):
        ''' Add a new guy to the pedigree'''
        self.nodes[indiv]=PedNode(indiv)

    def set_father(self,offspring,dad):
        '''Set that a dad -> offspring relationship

        If called more than once with a different dad a warning is
        raised and the first call takes precedence.  If dad and/or
        offspring are not in the pedigree they will be added.


        Parameters
        ----------
        offspring : object
            offspring to be set. Must be hashable.
        dad : object
           dad to be set. Must be hashable.
        '''
        try:
            indiv = self.nodes[offspring]
        except KeyError:
            self.add_node(offspring)
            indiv = self.nodes[offspring]

        try:
            father = self.nodes[dad]
        except KeyError:
            self.add_node(dad)
            father = self.nodes[dad]

        indiv.father = father
        father.add_child(indiv)
        self._reset_families()
        
    def get_father(self,indiv):
        try:
            member = self.nodes[indiv]
        except KeyError:
            raise KeyError('Individual not found in pedigree')
        return member.father
        
    def set_mother(self,offspring,mom):
        '''Set that a mom -> offspring relationship

        If called more than once with a different mom a warning is
        raised and the first call takes precedence.  If mom and/or
        offspring are not in the pedigree they will be added.

        Parameters
        ----------
        offspring : object
           offspring to be set. Must be hashable.
        mom : object
           mom to be set. Must be hashable.
        '''
        try:
            indiv = self.nodes[offspring]
        except KeyError:
            self.add_node(son)
            indiv = self.nodes[offspring]

        try:
            mother = self.nodes[mom]
        except KeyError:
            self.add_node(mom)
            mother = self.nodes[mom]

        indiv.mother = mother
        mother.add_child(indiv)
        self._reset_families()

    def get_mother(self,indiv):
        try:
            member = self.nodes[indiv]
        except KeyError:
            raise KeyError('Individual not found in pedigree')
        return member.mother

    
    def _get_relatives(self,node,connected = None):
        """
        Get all individuals connected to a node in the pedigree.

        Parameters
        ----------
        node : obj of class PedNode
           the investigated node
        connected : list of obj of class PedNode 
           the list of ancestors and descendants of node, populated by calling this method.

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
            self._get_relatives(off,connected)
        if node.father!=None:
            self._get_relatives(node.father,connected)
        if node.mother!=None:
            self._get_relatives(node.mother,connected)
        return connected

    @property
    def unrelated_individuals(self):
        """List of unrelated individuals with no parents and no child in the pedigree"""
        return [x.indiv for x in self.founders if len(x.children)==0]

    def build_families(self):
        '''
        Cluster individuals by families
        '''
        founders2go = self.founders[:]
        offspring2go = self.non_founders[:]
        families = []
        while len(founders2go) > 0:
            relatives = self._get_relatives( founders2go[0] )
            # print(f"Family of {founders2go[0]}")
            # print(f"[founders2go]: {[x.indiv for x in founders2go]}")
            # print(f"[offspring2go]: {[x.indiv for x in offspring2go]}")
            # print(f"[relatives]: {[(x.indiv,x) for x in relatives]}")
            fam = Pedigree.from_pednodes(relatives)
            families.append(fam)
            famFounders=[]
            famNonFounders=[]
            for x in relatives:
                if x in founders2go:
                    famFounders.append(x)
                else:
                    famNonFounders.append(x)
            # print(f"[famFounders]:  {[x.indiv for x in famFounders]}")
            # print(f"[famNonFounders]: {[x.indiv for x in famNonFounders]}")
            for x in famFounders:
                founders2go.remove(x)
            for x in famNonFounders:
                    offspring2go.remove(x)
        return families
