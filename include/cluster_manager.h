// --------------------------------------------------------------------------
//
// Copyright (C) 2018 by the ExWave authors
//
// This file is part of the ExWave library.
//
// The ExWave library is free software; you can use it, redistribute it,
// and/or modify it under the terms of the GNU Lesser General Public License
// as published by the Free Software Foundation; either version 2.1 of the
// License, or (at your option) any later version. The full text of the
// license can be found in the file LICENSE at the top level of the ExWave
// distribution.
//
// --------------------------------------------------------------------------

#ifndef cluster_manager_h_
#define cluster_manager_h_

#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>

namespace HDG_WE
{
  using namespace dealii;

  template <typename Number>
  class ClusterManager
  {
  public:

    static const unsigned int n_vect = VectorizedArray<Number>::n_array_elements;

    ClusterManager(const double rel_tol=1.e-4)
      :relative_tolerance(rel_tol)
    {}

    template <typename Operator> void perform_time_step(const Operator &op,
                                                        const LinearAlgebra::distributed::Vector<Number>  &src,
                                                        LinearAlgebra::distributed::Vector<Number>        &dst) const;

    template <int dim> void propose_cluster_categorization(std::vector<unsigned int> &categories,
                                                           const Triangulation<dim> &tria,
                                                           const std::vector<const DoFHandler<dim> *> &dof_handlers,
                                                           const unsigned int max_clusters,
                                                           const unsigned int max_diff,
                                                           const double cfl_number);

    template <int dim> void iterate_cluster_categorization(const Triangulation<dim> &tria,
                                                           const std::vector<const DoFHandler<dim> *> &dof_handlers,
                                                           const unsigned int max_clusters,
                                                           const unsigned int max_diff,
                                                           const double cfl_number);

    template <int dim> unsigned int cell_weight(const typename parallel::distributed::Triangulation<dim>::cell_iterator &cell,
                                                const typename parallel::distributed::Triangulation<dim>::CellStatus status)
    {
      const unsigned int cell_weight = element_categories[cell->active_cell_index()]/3;

      return cell_weight;
    }

    template <typename Operator> void setup(const Operator &op);

    template <typename Operator> void setup_mf_index_to_cell_index(const Operator &op);

    //{ routines for WaveEquationOperationADERLTS to ask for:
    Number get_cell_time_step(unsigned int cell) const
    {
      return cluster_timestepmultiples[cell_cluster_ids[cell]]*fastest_time_step;
    }

    bool is_evaluate_cell(unsigned int cell) const
    {
      return evaluate_cell[cell];
    }

    bool is_update_cell(unsigned int cell) const
    {
      return update_cell[cell];
    }

    bool is_evaluate_face(unsigned int face) const
    {
      return evaluate_face[face];
    }

    Number get_t1() const
    {
      return t1;
    }

    Number get_t2() const
    {
      return t2;
    }

    Number get_te(unsigned int cell) const
    {
      return cell_timelevels[cell];
    }

    Number get_dt() const
    {
      return dt;
    }

    std::bitset<VectorizedArray<Number>::n_array_elements> get_phi_to_dst(unsigned int face) const
    {
      return phi_to_dst[face];
    }

    std::bitset<VectorizedArray<Number>::n_array_elements> get_phi_to_fluxmemory(unsigned int face) const
    {
      return phi_to_fluxmemory[face];
    }

    std::bitset<VectorizedArray<Number>::n_array_elements> get_phi_neighbor_to_dst(unsigned int face) const
    {
      return phi_neighbor_to_dst[face];
    }

    std::bitset<VectorizedArray<Number>::n_array_elements> get_phi_neighbor_to_fluxmemory(unsigned int face) const
    {
      return phi_neighbor_to_fluxmemory[face];
    }
    //}

    // vector of length cells with correspondent cluster ids
    std::vector<unsigned int> cell_cluster_ids;

    // vector of length cells with correspondent cluster ids
    std::vector<unsigned int> element_categories;

    // vectors of length cells with indicator for neighbors with smaller/bigger time step size
    std::vector<bool> cell_have_faster_neighbor;
    std::vector<bool> cell_have_slower_neighbor;

    // number of clusters
    unsigned int n_clusters;

    // fastest time step
    Number fastest_time_step;

    unsigned int cluster_diff;

    // is flux considered
    mutable bool is_fluxmemory_considered;

    // improved gradient and divergence
    mutable LinearAlgebra::distributed::Vector<Number> improvedgraddiv;

    // state vector
    mutable LinearAlgebra::distributed::Vector<Number> state;

  private:
    // temporary state for reconstruction
    mutable LinearAlgebra::distributed::Vector<Number> temporary_recon_state;

    // vector of length clusters with correspondent time levels
    mutable std::vector<Number> cluster_timelevels;

    // vector of length cells with correspondent time levels
    mutable std::vector<Number> cell_timelevels;

    // vector of length clusters with correspondent time step sizes
    std::vector<int> cluster_timestepmultiples;

    // update order (of length updates)
    std::vector<unsigned int> cluster_update_order;

    // update times (of length updates)
    mutable std::vector<std::vector<Number> > cluster_update_times;

    // number of updates
    unsigned int n_updates;

    // number of macro cells (row+column)
    unsigned int n_cells_with_ghosts;

    // number of macro row cells
    unsigned int n_cells;

    // vector with flags for cell evaluation
    mutable std::vector<bool> evaluate_cell;

    // vector with flags for cell evaluation
    mutable std::vector<bool> update_cell;

    // vector with flags for face evaluation
    mutable std::vector<bool> evaluate_face;

    // evaluation times
    mutable Number t1,t2,te,dt;

    // helper times
    mutable Number t1fa, t2fa, t1sl, t2sl, t1sa, t2sa;

    // relative tolerance to check for evaluation
    const Number relative_tolerance;

    // mapping between indices of matrix free thing and the rest of the world
    std::vector<int> mf_index;

    // use ader post
    bool use_ader_post;

    // masks for each face
    mutable std::vector<std::bitset<n_vect> > phi_to_dst;
    mutable std::vector<std::bitset<n_vect> > phi_neighbor_to_dst;
    mutable std::vector<std::bitset<n_vect> > phi_to_fluxmemory;
    mutable std::vector<std::bitset<n_vect> > phi_neighbor_to_fluxmemory;

    // helper vectors to avoid the get_cell_iterator in update_elements
    std::vector<std::vector<std::bitset<n_vect> > > cell_neighbor_has_children; // order is n,e,v
    std::vector<std::vector<std::vector<int> > > cell_neighbor_index;           // order is n,v,e
    std::vector<std::vector<std::vector<int> > > cell_neighbor_active_cell_index; // order is n,v,e

    std::vector<std::vector<unsigned int> > mf_faceinfo_cellsminus;
    std::vector<std::vector<unsigned int> > mf_faceinfo_cellsplus;

    template <typename Operator> void update_elements(const Operator &op,
                                                      LinearAlgebra::distributed::Vector<Number>        &dst,
                                                      LinearAlgebra::distributed::Vector<Number>        &local_state,
                                                      const unsigned int actual_cluster,
                                                      const bool write_to_fluxmemory = true) const;
  };

}

#endif
