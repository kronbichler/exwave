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

#ifndef cluster_manager_templates_h_
#define cluster_manager_templates_h_

#include "cluster_manager.h"

namespace HDG_WE
{

  template <typename Number>
  template <int dim>
  void ClusterManager<Number>::iterate_cluster_categorization(const Triangulation<dim> &tria,
                                                              const std::vector<const DoFHandler<dim> *> &dof_handlers,
                                                              const unsigned int max_clusters,
                                                              const unsigned int max_diff,
                                                              const double cfl_number)
  {
    std::vector<unsigned int> temporary_cluster_ids(tria.n_active_cells(),0);
    IndexSet elerowset(tria.n_global_active_cells());
    IndexSet elecolset(tria.n_global_active_cells());
    const unsigned dofs_per_cell = dof_handlers[0]->get_fe().dofs_per_cell;
    std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);
    std::vector<int> contiguous_dof_index_for_cell(tria.n_global_active_cells(),-1);

    // setup maps
    {
      typename DoFHandler<dim>::active_cell_iterator cell = dof_handlers[0]->begin_active(), endc = dof_handlers[0]->end();
      for (; cell!=endc; ++cell)
        if (!cell->is_artificial())
          {
            cell->get_dof_indices(local_dof_indices);
            contiguous_dof_index_for_cell[cell->active_cell_index()] = local_dof_indices[0]/dofs_per_cell;
            if (!cell->is_ghost())
              elerowset.add_index(local_dof_indices[0]/dofs_per_cell);
            else
              elecolset.add_index(local_dof_indices[0]/dofs_per_cell);
          }
    }
    elerowset.compress();
    elecolset.compress();

    LinearAlgebra::distributed::Vector<Number> distributed_cell_categories;
    distributed_cell_categories.reinit(elerowset,elecolset,MPI_COMM_WORLD);

    typename Triangulation<dim>::active_cell_iterator cell = tria.begin_active(), endc = tria.end();
    Number minimaltimestep = std::numeric_limits<Number>::max();
    std::vector<Number> timestepcollection(tria.n_active_cells());
    for (; cell!=endc; ++cell)//if (cell->is_locally_owned())
      if (!cell->is_artificial())
        {
          timestepcollection[cell->active_cell_index()] = cfl_number * cell->minimum_vertex_distance();
          if (timestepcollection[cell->active_cell_index()]<minimaltimestep)
            minimaltimestep = timestepcollection[cell->active_cell_index()];
        }
    minimaltimestep = Utilities::MPI::min(minimaltimestep,MPI_COMM_WORLD);

    Number eps = minimaltimestep/10.0; // we want at least one cell with category 0 but sometimes int(double/double) does give 0 and sometimes 1 -> therefore eps
    n_clusters = 0;
    for (unsigned int ac=0; ac<tria.n_active_cells(); ++ac)
      {
        temporary_cluster_ids[ac] = int((timestepcollection[ac]-eps)/minimaltimestep);
        if (temporary_cluster_ids[ac]>n_clusters)
          n_clusters = temporary_cluster_ids[ac];
      }
    n_clusters++; // number of clusters is one more than highest cluster id
    n_clusters = Utilities::MPI::max(n_clusters,MPI_COMM_WORLD);

    // here, we already have to find a valid clustering, not later!
    cluster_diff = 1;
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout<<"ncluster vorher "<<n_clusters<<std::endl;
    if (n_clusters>max_clusters)
      {
        cluster_diff = n_clusters/max_clusters+1;
        cluster_diff = std::min(cluster_diff,max_diff);
        for (unsigned int ac=0; ac<tria.n_active_cells(); ++ac)
          temporary_cluster_ids[ac] = temporary_cluster_ids[ac]/cluster_diff;
      }
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout<<"actual cluster diff "<<cluster_diff<<std::endl;

    cell_have_faster_neighbor.resize(tria.n_active_cells());
    cell_have_slower_neighbor.resize(tria.n_active_cells());

    bool faster_and_slower_neighbor = true;
    unsigned int count = 0;
    while (faster_and_slower_neighbor && count<100)
      {
        count++;
        cell_have_faster_neighbor.clear();
        cell_have_slower_neighbor.clear();
        for (unsigned int c=0; c<n_clusters; ++c)
          {
            typename Triangulation<dim>::active_cell_iterator cell = tria.begin_active(),endc = tria.end();
            for (; cell!=endc; ++cell)
              if (!cell->is_artificial() && !cell->is_ghost())
                {
                  if (temporary_cluster_ids[cell->active_cell_index()] == c)
                    {
                      bool is_anyone_faster = false;
                      for (unsigned int n=0; n<GeometryInfo<dim>::faces_per_cell; ++n)
                        if (cell->neighbor_index(n)>=0)
                          {
                            if (cell->neighbor(n)->active() && !cell->neighbor(n)->is_artificial() && cell->neighbor(n)->has_children())
                              {
                                for (unsigned int subfaces = 0; subfaces < GeometryInfo<dim>::max_children_per_face; ++subfaces)
                                  {
                                    if (temporary_cluster_ids[cell->neighbor_child_on_subface(n,subfaces)->active_cell_index()]<c)
                                      is_anyone_faster = true;
                                  }
                              }
                            else if (cell->neighbor(n)->active() &&  !cell->neighbor(n)->is_artificial() &&  temporary_cluster_ids[cell->neighbor(n)->active_cell_index()]<c )
                              is_anyone_faster = true;
                          }
                      for (unsigned int n=0; n<GeometryInfo<dim>::faces_per_cell; ++n)
                        {
                          if (cell->neighbor_index(n)>=0)
                            {
                              if ( cell->neighbor(n)->active() &&  !cell->neighbor(n)->is_artificial() && cell->neighbor(n)->has_children() )
                                {
                                  for (unsigned int subfaces = 0; subfaces < GeometryInfo<dim>::max_children_per_face; ++subfaces)
                                    {
                                      if (temporary_cluster_ids[cell->neighbor_child_on_subface(n,subfaces)->active_cell_index()]>c)
                                        {
                                          if (is_anyone_faster)
                                            temporary_cluster_ids[cell->neighbor_child_on_subface(n,subfaces)->active_cell_index()] = c;
                                          else
                                            temporary_cluster_ids[cell->neighbor_child_on_subface(n,subfaces)->active_cell_index()] = c+1;
                                        }
                                    }
                                }
                              else if (cell->neighbor(n)->active() && !cell->neighbor(n)->is_artificial())
                                {
                                  if (temporary_cluster_ids[cell->neighbor(n)->active_cell_index()]>c)
                                    {
                                      if (is_anyone_faster)
                                        temporary_cluster_ids[cell->neighbor(n)->active_cell_index()] = c;
                                      else
                                        temporary_cluster_ids[cell->neighbor(n)->active_cell_index()] = c+1;
                                    }
                                }
                            }
                        }
                    } // if(temporary_cluster_ids[cell->active_cell_index()] == c)
                } // for (; cell!=endc; ++cell)
          } // for (unsigned int c=0; c<n_clusters; ++c)

        // communicate cluster ids now
        {
          distributed_cell_categories = 0.;
          typename Triangulation<dim>::active_cell_iterator cell = tria.begin_active(),endc = tria.end();
          for (; cell!=endc; ++cell)
            if (!cell->is_artificial()) // && !cell->is_ghost())
              {
                distributed_cell_categories[contiguous_dof_index_for_cell[cell->active_cell_index()]] = temporary_cluster_ids[cell->active_cell_index()];
              }

          distributed_cell_categories.compress(VectorOperation::min); // minimize
          distributed_cell_categories.update_ghost_values();

          // and bring back
          cell = tria.begin_active();
          for (; cell!=endc; ++cell)
            if (!cell->is_artificial())
              {
                temporary_cluster_ids[cell->active_cell_index()] = distributed_cell_categories[contiguous_dof_index_for_cell[cell->active_cell_index()]];
              }
        }
        // determine if a cell has a faster neighbor
        cell_have_faster_neighbor.resize(tria.n_active_cells(),false);
        cell_have_slower_neighbor.resize(tria.n_active_cells(),false);

        faster_and_slower_neighbor = false;
        typename Triangulation<dim>::active_cell_iterator cell = tria.begin_active(),endc = tria.end();
        for (; cell!=endc; ++cell)
          if (!cell->is_artificial()) // && !cell->is_ghost())
            {
              unsigned int current_c = temporary_cluster_ids[cell->active_cell_index()];
              bool is_anyone_faster = false;
              bool is_anyone_slower = false;
              for (unsigned int n=0; n<GeometryInfo<dim>::faces_per_cell; ++n)
                {
                  if (cell->neighbor_index(n)>=0)
                    {
                      if (cell->neighbor(n)->active() &&  !cell->neighbor(n)->is_artificial() &&  cell->neighbor(n)->has_children() )
                        {
                          for (unsigned int subfaces = 0; subfaces < GeometryInfo<dim>::max_children_per_face; ++subfaces)
                            {
                              if (temporary_cluster_ids[cell->neighbor_child_on_subface(n,subfaces)->active_cell_index()]<current_c)
                                {
                                  is_anyone_faster = true;
                                  cell_have_faster_neighbor[cell->active_cell_index()] = true;
                                }
                              else if (temporary_cluster_ids[cell->neighbor_child_on_subface(n,subfaces)->active_cell_index()]>current_c)
                                {
                                  is_anyone_slower = true;
                                  cell_have_slower_neighbor[cell->active_cell_index()] = true;
                                }
                              if (int(temporary_cluster_ids[cell->neighbor_child_on_subface(n,subfaces)->active_cell_index()])<int(current_c-1)
                                  ||  temporary_cluster_ids[cell->neighbor_child_on_subface(n,subfaces)->active_cell_index()]>current_c+1)
                                {
                                  faster_and_slower_neighbor = true;
                                }
                            }
                        }
                      else if (cell->neighbor(n)->active() && !cell->neighbor(n)->is_artificial())
                        {
                          if (temporary_cluster_ids[cell->neighbor(n)->active_cell_index()]<current_c)
                            {
                              is_anyone_faster = true;
                              cell_have_faster_neighbor[cell->active_cell_index()] = true;
                            }
                          else if (temporary_cluster_ids[cell->neighbor(n)->active_cell_index()]>current_c)
                            {
                              is_anyone_slower = true;
                              cell_have_slower_neighbor[cell->active_cell_index()] = true;
                            }
                          if (int(temporary_cluster_ids[cell->neighbor(n)->active_cell_index()])<int(current_c-1)
                              ||  temporary_cluster_ids[cell->neighbor(n)->active_cell_index()]>current_c+1)
                            {
                              faster_and_slower_neighbor = true;
                            }
                        }
                    }
                }
              if (is_anyone_faster&&is_anyone_slower)
                {
                  faster_and_slower_neighbor = true;
                }
            } // for (; cell!=endc; ++cell)
        faster_and_slower_neighbor = Utilities::MPI::max(double(faster_and_slower_neighbor),MPI_COMM_WORLD);
      } // while (faster_and_slower_neighbor && count<100)

    if (count==100)
      Assert(false,ExcMessage("could not find suited categories in 100 iterations"));
    std::cout<<"needed to iterate "<<count<<" times to get valid categories"<<std::endl;

    element_categories.resize(tria.n_active_cells());
    n_clusters = 0;
    for (unsigned int ac=0; ac<tria.n_active_cells(); ++ac)
      {
        element_categories[ac] = 3*temporary_cluster_ids[ac]+1*int(cell_have_slower_neighbor[ac]) + 2*int(cell_have_faster_neighbor[ac]); // build category indicating not only cluster but also if it has faster or slower neighbor
        if (temporary_cluster_ids[ac]>n_clusters)
          n_clusters = temporary_cluster_ids[ac]; // reset n_clusters
      }
    n_clusters++;
    n_clusters = Utilities::MPI::max(n_clusters,MPI_COMM_WORLD);
    if (!Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
      std::cout<<"final number of clusters "<<n_clusters<<std::endl;
  }


  template <typename Number>
  template <int dim>
  void ClusterManager<Number>::propose_cluster_categorization(std::vector<unsigned int> &categories,
                                                              const Triangulation<dim> &tria,
                                                              const std::vector<const DoFHandler<dim> *> &dof_handlers,
                                                              const unsigned int max_clusters,
                                                              const unsigned int max_diff,
                                                              const double cfl_number)
  {
    this->iterate_cluster_categorization(tria,dof_handlers,max_clusters,max_diff,cfl_number);

    // cast triangulation
    parallel::distributed::Triangulation<dim> *triapll = (dynamic_cast<parallel::distributed::Triangulation<dim>*>
                                                          (const_cast<dealii::Triangulation<dim>*>
                                                           (&tria)));
    // setup weights of cells for processors according to clusters
    // triapll->signals.cell_weight.connect([&] (const typename parallel::distributed::Triangulation<dim>::cell_iterator &cell,
    //                                          const typename parallel::distributed::Triangulation<dim>::CellStatus ) -> unsigned int
    // { return (n_clusters-int(element_categories[cell->active_cell_index()]/3)-1)*cluster_diff*1000; });
    
    // repartition triangulation
    //triapll->repartition();
    
    // tell all the dof handlers what happened
    for (unsigned int i=0; i<dof_handlers.size(); ++i)
      (const_cast<dealii::DoFHandler<dim> *>(dof_handlers[i]))->distribute_dofs(dof_handlers[i]->get_fe());

    // setup the cluster proposition according to the new cell distribution - don't try anything too smart, just do it again
    this->iterate_cluster_categorization(tria,dof_handlers,max_clusters,max_diff,cfl_number);

    // bring in shape and fill
    categories.resize(tria.n_active_cells());
    for (unsigned int ac=0; ac<tria.n_active_cells(); ++ac)
      categories[ac] = element_categories[ac];

    return;
  }


  template <typename Number>
  template <typename Operator>
  void ClusterManager<Number>::setup_mf_index_to_cell_index(const Operator &op)
  {
    mf_index.resize(op.get_matrix_free().get_dof_handler(0).get_triangulation().n_active_cells());

    for (unsigned int i=0; i<n_cells_with_ghosts; ++i)
      for (unsigned int v=0; v<op.get_matrix_free().n_components_filled(i); ++v)
        mf_index[op.get_matrix_free().get_cell_iterator(i,v)->active_cell_index()] = i;
  }

  template <typename Number>
  template <typename Operator>
  void ClusterManager<Number>::setup(const Operator &op)
  {
    use_ader_post = op.parameters.use_ader_post;

    // the input value for dt is the smallest allowed time step, all other time
    // steps now must be set to multiples of this dt
    fastest_time_step = op.get_time_control().get_time_step();

    n_cells_with_ghosts = op.get_matrix_free().n_macro_cells()+op.get_matrix_free().n_ghost_cell_batches();
    n_cells = op.get_matrix_free().n_macro_cells();

    // setup matrix free index to cell index
    setup_mf_index_to_cell_index(op);

    // setup everything in matrixfree layout
    cell_have_slower_neighbor.clear();
    cell_have_faster_neighbor.clear();
    cell_have_slower_neighbor.resize(n_cells_with_ghosts);
    cell_have_faster_neighbor.resize(n_cells_with_ghosts);
    cell_cluster_ids.resize(n_cells_with_ghosts);

    // read the cluster categorization
    for (unsigned int cell=0; cell<n_cells_with_ghosts; ++cell)
      for (unsigned int v=0; v<op.get_matrix_free().n_components_filled(cell); ++v)
        {
          unsigned int index = op.get_matrix_free().get_cell_iterator(cell,v)->active_cell_index();
          cell_cluster_ids[cell] = int(element_categories[index]/3); // use the categories to determine cluster and faster and slower
          if (element_categories[index]%3==2)
            cell_have_faster_neighbor[cell] = true;
          if (element_categories[index]%3==1)
            cell_have_slower_neighbor[cell] = true;

        }

    // next stuff to init
    cluster_timelevels.clear();
    cell_timelevels.clear();
    evaluate_cell.clear();
    update_cell.clear();
    evaluate_face.clear();
    phi_to_dst.clear();
    phi_neighbor_to_dst.clear();
    phi_to_fluxmemory.clear();
    phi_neighbor_to_fluxmemory.clear();
    cluster_timelevels.resize(n_clusters,op.get_time_control().get_time());
    cell_timelevels.resize(n_cells_with_ghosts,op.get_time_control().get_time());
    evaluate_cell.resize(n_cells_with_ghosts,false);
    update_cell.resize(n_cells_with_ghosts,false);
    evaluate_face.resize(op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches(),false);
    phi_to_dst.resize(op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches());
    phi_neighbor_to_dst.resize(op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches());
    phi_to_fluxmemory.resize(op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches());
    phi_neighbor_to_fluxmemory.resize(op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches());

    // set the cluster time steps
    cluster_timestepmultiples.resize(n_clusters);
    for (unsigned c=0; c<n_clusters; ++c)
      cluster_timestepmultiples[c] = cluster_diff*int(c)+1;

    // some statistics
    std::vector<unsigned int> num_cell_cluster(n_clusters,0);
    for (unsigned int i=0; i<n_cells; ++i)
      num_cell_cluster[cell_cluster_ids[i]]++;
    Utilities::MPI::sum(num_cell_cluster,MPI_COMM_WORLD,num_cell_cluster);

    // for curiosity
    if (!Utilities::MPI::this_mpi_process(MPI_COMM_WORLD))
      {
        for (unsigned c=0; c<n_clusters; ++c)
          std::cout<<"cluster  "<<c<<" contains "<<num_cell_cluster[c]<<" cells and works with time step "<<cluster_timestepmultiples[c]*fastest_time_step<<std::endl;

        std::cout<<"cluster_timestepmultiples"<<std::endl;
        for (unsigned c=0; c<n_clusters; ++c)
          std::cout<<cluster_timestepmultiples[c]<<std::endl;
        std::cout<<"n_clusters "<<n_clusters<<" clusterdiff "<<cluster_diff<<" levelmax "<<cluster_diff*(n_clusters-1)+1<<std::endl;
      }

    // determine how one updates from one global time step to the next
    // for a general clustering, we have to check the update criteria for each cluster
    // restriction is: neighboring clusters are only one apart
    std::vector<unsigned int> templevels(n_clusters);
    std::vector<double> t_a_b(2,0.0);

    // need this clear for adaptivity
    cluster_update_times.clear();
    cluster_update_order.clear();
    unsigned int max_level = cluster_diff*(n_clusters-1)+1;
    while (*std::min_element(templevels.begin(),templevels.end())<max_level)
      {
        for (unsigned c=0; c<n_clusters; ++c)
          {
            if (templevels[c]<cluster_diff*(n_clusters-1)+1)
              {
                if (c==0)
                  {
                    if (std::min(max_level,templevels[c]+cluster_timestepmultiples[c])<=std::min(max_level,templevels[c+1]+cluster_timestepmultiples[c+1]))
                      {
                        t_a_b[0] = templevels[c]*fastest_time_step;
                        templevels[c]+=cluster_timestepmultiples[c];
                        cluster_update_order.push_back(c);
                        t_a_b[1] = templevels[c]*fastest_time_step;
                        cluster_update_times.push_back(t_a_b);
                      }
                  }
                else if (c==n_clusters-1)
                  {
                    if (std::min(max_level,templevels[c]+cluster_timestepmultiples[c])<=std::min(max_level,templevels[c-1]+cluster_timestepmultiples[c-1]))
                      {
                        t_a_b[0] = templevels[c]*fastest_time_step;
                        templevels[c]+=cluster_timestepmultiples[c];
                        cluster_update_order.push_back(c);
                        t_a_b[1] = templevels[c]*fastest_time_step;
                        cluster_update_times.push_back(t_a_b);
                      }
                  }
                else
                  {
                    if ( std::min(max_level,templevels[c]+cluster_timestepmultiples[c])<=std::min(max_level,templevels[c+1]+cluster_timestepmultiples[c+1])
                         &&
                         std::min(max_level,templevels[c]+cluster_timestepmultiples[c])<=std::min(max_level,templevels[c-1]+cluster_timestepmultiples[c-1]) )
                      {
                        t_a_b[0] = templevels[c]*fastest_time_step;
                        templevels[c]+=cluster_timestepmultiples[c];
                        cluster_update_order.push_back(c);
                        t_a_b[1] = templevels[c]*fastest_time_step;
                        cluster_update_times.push_back(t_a_b);
                      }
                  }
              }

          }
      }

    // correct for too long updates
    for (unsigned c=0; c<n_clusters; ++c)
      if (templevels[c]>cluster_diff*(n_clusters-1)+1)
        {
          templevels[c] = cluster_diff*(n_clusters-1)+1;
        }

    // tell the wave equation problem with which maximum time step we update
    op.get_time_control().set_time_step(fastest_time_step*(cluster_diff*(n_clusters-1)+1));

    n_updates = cluster_update_order.size();

    for (unsigned c=0; c<cluster_update_times.size(); ++c)
      if (cluster_update_times[c][1]>(cluster_diff*(n_clusters-1)+1)*fastest_time_step)
        cluster_update_times[c][1] = (cluster_diff*(n_clusters-1)+1)*fastest_time_step;

    for (unsigned c=0; c<cluster_update_times.size(); ++c)
      {
        cluster_update_times[c][0] += op.get_time_control().get_time();
        cluster_update_times[c][1] += op.get_time_control().get_time();
      }

    // for curiosity
    //for(unsigned c=0; c<cluster_update_order.size(); ++c)
    //  std::cout<<"c "<<c<<" updatecluster "<<cluster_update_order[c]<<std::endl;

    // minimizing the calls to get_cell_iterator in update_cells by storing respective indices beforehand
    cell_neighbor_has_children.clear();
    cell_neighbor_index.clear();
    cell_neighbor_active_cell_index.clear();

    cell_neighbor_has_children.resize(GeometryInfo<Operator::dimension>::faces_per_cell);
    cell_neighbor_index.resize(GeometryInfo<Operator::dimension>::faces_per_cell);
    cell_neighbor_active_cell_index.resize(GeometryInfo<Operator::dimension>::faces_per_cell);
    for (unsigned n=0; n<GeometryInfo<Operator::dimension>::faces_per_cell; ++n)
      {
        cell_neighbor_has_children[n].resize(n_cells_with_ghosts);
        cell_neighbor_index[n].resize(n_vect);
        cell_neighbor_active_cell_index[n].resize(n_vect);
        for (unsigned v=0; v<n_vect; ++v)
          {
            cell_neighbor_index[n][v].resize(n_cells_with_ghosts);
            cell_neighbor_active_cell_index[n][v].resize(n_cells_with_ghosts);
          }
      }

    for (unsigned int n=0; n<GeometryInfo<Operator::dimension>::faces_per_cell; ++n)
      for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
        for (unsigned int v=0; v<op.get_matrix_free().n_components_filled(e); ++v)
          {
            cell_neighbor_index[n][v][e] = op.get_matrix_free().get_cell_iterator(e,v)->neighbor_index(n);
            if (cell_neighbor_index[n][v][e] >= 0 && op.get_matrix_free().get_cell_iterator(e,v)->neighbor(n)->active() )
              cell_neighbor_active_cell_index[n][v][e] = op.get_matrix_free().get_cell_iterator(e,v)->neighbor(n)->active_cell_index();
          }

    for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
      for (unsigned int n=0; n<GeometryInfo<Operator::dimension>::faces_per_cell; ++n)
        for (unsigned int v=0; v<op.get_matrix_free().n_components_filled(e); ++v)
          if (cell_neighbor_index[n][v][e] >= 0)
            cell_neighbor_has_children[n][e][v] = op.get_matrix_free().get_cell_iterator(e,v)->neighbor(n)->has_children();

    mf_faceinfo_cellsminus.clear();
    mf_faceinfo_cellsplus.clear();
    mf_faceinfo_cellsminus.resize(n_vect);
    mf_faceinfo_cellsplus.resize(n_vect);
    for (unsigned int v=0; v<n_vect; ++v)
      {
        mf_faceinfo_cellsminus[v].resize(op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches());
        mf_faceinfo_cellsplus[v].resize(op.get_matrix_free().n_inner_face_batches());
      }

    for (unsigned int f=0; f<op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches(); ++f)
      for (unsigned int v=0; v<n_vect; ++v)
        {
          mf_faceinfo_cellsminus[v][f] = op.get_matrix_free().get_face_info(f).cells_interior[v];
          if (f<op.get_matrix_free().n_inner_face_batches())
            mf_faceinfo_cellsplus[v][f] = op.get_matrix_free().get_face_info(f).cells_exterior[v];
        }
  }



  template <typename Number>
  template <typename Operator>
  void ClusterManager<Number>::perform_time_step(const Operator &op,
                                                 const LinearAlgebra::distributed::Vector<Number>  &src,
                                                 LinearAlgebra::distributed::Vector<Number>        &dst) const
  {
    state = src;
    dst = 0;

    if (use_ader_post)
      {
        // standard reconstruction:
        op.apply(state,improvedgraddiv);
      }

    // update cell times
    for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
      cell_timelevels[e] = cluster_timelevels[cell_cluster_ids[e]];

    for (unsigned int cycle = 0; cycle<n_updates; ++cycle)
      {
        // we want to update the elements of cluster cluster_update_order[cycle]
        unsigned int actual_cluster = cluster_update_order[cycle];
        double actual_cluster_time = cluster_timelevels[actual_cluster];

        // security check
        if (actual_cluster_time!=cluster_update_times[cycle][0])
          {
            std::cout<<"actual cluster time "<<actual_cluster_time<<" cluster update times "<<cluster_update_times[cycle][0]<<std::endl;
            std::cout<<"cycle "<<cycle<<" actual cluster "<<actual_cluster<<std::endl;
            Assert(false,ExcMessage("cluster time missmatch"));
          }
        is_fluxmemory_considered = true;

        for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
          {
            if (cell_cluster_ids[e]==actual_cluster)
              update_cell[e] = true;
            else
              update_cell[e] = false;
          }

        double faster_cluster_time = 0.0;
        if (actual_cluster>0)
          {
            faster_cluster_time = cluster_timelevels[actual_cluster-1];
            t1fa = std::max(faster_cluster_time,actual_cluster_time);
            t2fa = std::min(faster_cluster_time+cluster_timestepmultiples[actual_cluster-1]*fastest_time_step,
                            actual_cluster_time+cluster_timestepmultiples[actual_cluster]*fastest_time_step);
            t2fa = std::min(t2fa,op.get_time_control().get_time());
          }
        else
          {
            t1fa = 0.0;
            t2fa = 0.0;
          }

        if (actual_cluster<n_clusters-1)
          {
            double slower_cluster_time = cluster_timelevels[actual_cluster+1];
            t1sl = std::max(slower_cluster_time,actual_cluster_time);
            t2sl = std::min(slower_cluster_time+cluster_timestepmultiples[actual_cluster+1]*fastest_time_step,
                            actual_cluster_time+cluster_timestepmultiples[actual_cluster]*fastest_time_step);
            t2sl = std::min(t2sl,op.get_time_control().get_time());
          }
        else
          {
            t1sl = 0.0;
            t2sl = 0.0;
          }
        t1sa = actual_cluster_time;
        t2sa = std::min(actual_cluster_time+cluster_timestepmultiples[actual_cluster]*fastest_time_step,op.get_time_control().get_time());

        dt = std::min(cluster_timestepmultiples[actual_cluster]*fastest_time_step,
                      op.get_time_control().get_time()-cluster_timelevels[actual_cluster]);

        update_elements(op,dst,state,actual_cluster);

        // update time level of cluster
        cluster_timelevels[actual_cluster] = cluster_update_times[cycle][1];

        // update cell times
        for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
          cell_timelevels[e] = cluster_timelevels[cell_cluster_ids[e]];

        // in case we want superconvergence, we need to do the reconstruction step as explained in the paper
        if (use_ader_post)
          {
            // init
            is_fluxmemory_considered = false;
            temporary_recon_state = state;

            // current cluster advanced in time
            actual_cluster_time = cluster_timelevels[actual_cluster];

            // which cells do we have to update to be able to perform reconstruction
            // all cells who are neighbor of actual cluster and  have lower or higher time level
            // start with faster cluster:
            std::vector<bool> temp_faster(n_cells_with_ghosts);
            std::vector<bool> temp_slower(n_cells_with_ghosts);
            if (actual_cluster>0)
              if (std::abs(actual_cluster_time-cluster_timelevels[actual_cluster-1])>relative_tolerance*fastest_time_step)
                {
                  for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
                    {
                      if (cell_cluster_ids[e]==actual_cluster-1 && cell_have_slower_neighbor[e])
                        temp_faster[e] = true;
                      else
                        temp_faster[e] = false;
                    }
                  // expand this selection by the neighbors of those already set (later we need neighbor of neighbor info!)
                  for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
                    update_cell[e]=false;
                  for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
                    {
                      for (unsigned int v=0; v<op.get_matrix_free().n_components_filled(e); ++v)
                        for (unsigned int n=0; n<GeometryInfo<Operator::dimension>::faces_per_cell; ++n)
                          if (cell_neighbor_index[n][v][e]>=0)
                            {
                              if (temp_faster[mf_index[cell_neighbor_active_cell_index[n][v][e]]] && cell_cluster_ids[e]==actual_cluster-1)
                                update_cell[e]=true;
                              else
                                update_cell[e]=false;
                            }
                    }
                  // combine both
                  for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
                    update_cell[e] = update_cell[e] || temp_faster[e];
                  temp_faster = update_cell;

                  t1fa = cluster_timelevels[actual_cluster-1];
                  t2fa = actual_cluster_time;

                  t1sl = t1fa;
                  t2sl = t2fa;

                  t1sa = t1fa;
                  t2sa = t2fa;

                  dt = t2fa-t1fa;

                  update_elements(op,dst,temporary_recon_state,actual_cluster-1,false);
                }

            // slower cluster
            if (actual_cluster<n_clusters-1)
              if (std::abs(actual_cluster_time-cluster_timelevels[actual_cluster+1])>relative_tolerance*fastest_time_step)
                {
                  for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
                    {
                      if (cell_cluster_ids[e]==actual_cluster+1 && cell_have_faster_neighbor[e])
                        temp_slower[e] = true;
                      else
                        temp_slower[e] = false;
                    }

                  // expand this selection by the neighbors of those already set (later we need neighbor of neighbor info!)
                  for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
                    update_cell[e] = false;
                  for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
                    {
                      for (unsigned int v=0; v<op.get_matrix_free().n_components_filled(e); ++v)
                        for (unsigned int n=0; n<GeometryInfo<Operator::dimension>::faces_per_cell; ++n)
                          if (cell_neighbor_index[n][v][e]>=0)
                            if (temp_slower[mf_index[cell_neighbor_active_cell_index[n][v][e]]] && cell_cluster_ids[e]==actual_cluster+1)
                              {
                                update_cell[e]=true;
                              }
                    }

                  // combine both
                  for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
                    update_cell[e] = (update_cell[e] || temp_slower[e]);
                  temp_slower = update_cell;

                  t1fa = cluster_timelevels[actual_cluster+1];
                  t2fa = actual_cluster_time;

                  t1sl = t1fa;
                  t2sl = t2fa;

                  t1sa = t1fa;
                  t2sa = t2fa;

                  dt = t2fa-t1fa;

                  update_elements(op,dst,temporary_recon_state,actual_cluster+1,false);
                }
            // now, the neighboring elements are at the required time level

            // reconstruction:
            for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
              {
                if (cell_cluster_ids[e]==actual_cluster)
                  evaluate_cell[e] = true;
                else
                  evaluate_cell[e] = false;
              }

            for (unsigned int f=0; f<op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches(); ++f)
              {
                evaluate_face[f] = false;
                phi_to_dst[f] = std::bitset<Operator::n_vect>(false);
                phi_neighbor_to_dst[f] = std::bitset<Operator::n_vect>(false);
                if (f<op.get_matrix_free().n_inner_face_batches())
                  {
                    for (unsigned int v=0; v<Operator::n_vect; ++v)
                      {
                        if (mf_faceinfo_cellsminus[v][f]!=numbers::invalid_unsigned_int && mf_faceinfo_cellsplus[v][f]!=numbers::invalid_unsigned_int)
                          if (evaluate_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect]
                              || evaluate_cell[mf_faceinfo_cellsplus[v][f]/Operator::n_vect])
                            {
                              evaluate_face[f] = true;
                              // set masks
                              if (evaluate_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect])
                                phi_to_dst[f][v] = true;
                              if (evaluate_cell[mf_faceinfo_cellsplus[v][f]/Operator::n_vect])
                                phi_neighbor_to_dst[f][v] = true;
                            }
                      }
                  }
                else
                  {
                    for (unsigned int v=0; v<Operator::n_vect; ++v)
                      {
                        if (mf_faceinfo_cellsminus[v][f]!=numbers::invalid_unsigned_int)
                          if (evaluate_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect])
                            {
                              evaluate_face[f] = true;
                              phi_to_dst[f][v] = true;
                            }
                      }
                  }
              }
            op.reconstruct_div_grad(temporary_recon_state,improvedgraddiv);
          }
      } // for(unsigned int cycle = 0; cycle<n_updates; ++cycle)

    // update the update times
    for (unsigned c=0; c<cluster_update_times.size(); ++c)
      for (unsigned int n=0; n<2 ; ++n)
        cluster_update_times[c][n] += op.get_time_control().get_time_step();

    // tell the time integrator about the new state
    dst = state;
  }



  template <typename Number>
  template <typename Operator>
  void ClusterManager<Number>::update_elements(const Operator &op,
                                               LinearAlgebra::distributed::Vector<Number> &dst,
                                               LinearAlgebra::distributed::Vector<Number> &local_state,
                                               const unsigned int actual_cluster,
                                               const bool write_to_fluxmemory) const
  {
    double actual_cluster_time = cluster_timelevels[actual_cluster];

    // security  checks
    for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
      {
        if (update_cell[e] == true && cell_cluster_ids[e]!=actual_cluster)
          Assert(false,ExcMessage("it is not allowed to call update elements on cells of differing cluster!"));
        if (update_cell[e] == true)
          if (cell_timelevels[e] != actual_cluster_time)
            Assert(false,ExcMessage("it is not allowed to call update elements on cells of differing time level!"));
      }

    // fill neighbor vector
    std::vector<bool> is_neighbor_of_update_cell(n_cells_with_ghosts,false);
    for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
      {
        if (update_cell[e] == false)
          {
            for (unsigned int v=0; v<op.get_matrix_free().n_components_filled(e); ++v)
              for (unsigned int n=0; n<GeometryInfo<Operator::dimension>::faces_per_cell; ++n)
                if (cell_neighbor_index[n][v][e]>=0)
                  {
                    if (cell_neighbor_has_children[n][e][v])
                      {
                        for (unsigned int subfaces = 0; subfaces < GeometryInfo<Operator::dimension>::max_children_per_face; ++subfaces)
                          {
                            if (update_cell[mf_index[op.get_matrix_free().get_cell_iterator(e,v)->neighbor_child_on_subface(n,subfaces)->active_cell_index()]])
                              is_neighbor_of_update_cell[e] = true;
                          }
                      }
                    else
                      {
                        if (update_cell[mf_index[cell_neighbor_active_cell_index[n][v][e]]])
                          is_neighbor_of_update_cell[e] = true;
                      }
                  }
          }
      }

    // contribution from the faster cluster
    if (actual_cluster>0)
      {
        // prepare everything for face evaluation between actual_cluster and actual_cluster-1
        t1 = t1fa;
        t2 = t2fa;

        if (std::abs(t2-t1)>relative_tolerance*fastest_time_step) // only do this if necessary
          {
            // setup the element list that need evaluation (all of update_cell who have faster neighbor
            // and all of actual_cluster-1 who are neighbor of update_cell)
            for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
              {
                if (cell_cluster_ids[e]==actual_cluster && cell_have_faster_neighbor[e] && update_cell[e])
                  evaluate_cell[e] = true;
                else if (cell_cluster_ids[e]==actual_cluster-1 && cell_have_slower_neighbor[e] && is_neighbor_of_update_cell[e])
                  evaluate_cell[e] = true;
                else
                  evaluate_cell[e] = false;
              }

            // setup the face list that need evaluation (if face is in between update cell and faster cluster)
            // also, setup the masks for the write functions
            for (unsigned int f=0; f<op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches(); ++f)
              {
                evaluate_face[f] = false;
                phi_to_dst[f] = std::bitset<Operator::n_vect>(false);
                phi_neighbor_to_dst[f] = std::bitset<Operator::n_vect>(false);
                phi_to_fluxmemory[f] = std::bitset<Operator::n_vect>(false);
                phi_neighbor_to_fluxmemory[f] = std::bitset<Operator::n_vect>(false);
                // inner faces
                if (f<op.get_matrix_free().n_inner_face_batches())
                  {
                    for (unsigned int v=0; v<Operator::n_vect; ++v)
                      if (mf_faceinfo_cellsminus[v][f]!=numbers::invalid_unsigned_int && mf_faceinfo_cellsplus[v][f]!=numbers::invalid_unsigned_int)
                        if (evaluate_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect] && evaluate_cell[mf_faceinfo_cellsplus[v][f]/Operator::n_vect])
                          {
                            // set face evaluation
                            if (  ( cell_cluster_ids[mf_faceinfo_cellsminus[v][f]/Operator::n_vect]==actual_cluster
                                    && cell_cluster_ids[mf_faceinfo_cellsplus[v][f]/Operator::n_vect]!=actual_cluster
                                    && is_neighbor_of_update_cell[mf_faceinfo_cellsplus[v][f]/Operator::n_vect] )
                                  ||
                                  ( cell_cluster_ids[mf_faceinfo_cellsminus[v][f]/Operator::n_vect]!=actual_cluster
                                    && cell_cluster_ids[mf_faceinfo_cellsplus[v][f]/Operator::n_vect]==actual_cluster
                                    && is_neighbor_of_update_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect] ) )
                              {
                                evaluate_face[f] = true;
                                // set masks
                                if (cell_cluster_ids[mf_faceinfo_cellsminus[v][f]/Operator::n_vect]==actual_cluster)
                                  {
                                    phi_to_dst[f][v] = true;
                                    if (write_to_fluxmemory)
                                      phi_neighbor_to_fluxmemory[f][v] = true;
                                  }
                                else
                                  {
                                    if (write_to_fluxmemory)
                                      phi_to_fluxmemory[f][v] = true;
                                    phi_neighbor_to_dst[f][v] = true;
                                  }
                              }
                          }
                  }
                else // boundary faces
                  {
                    // no contribution from boundary faces in this stage
                  }
              }
            // do first ader
            op.evaluate_cells_and_faces_first_ader(local_state,dst);
          }
      }

    // contribution from the slower cluster
    if (actual_cluster<n_clusters-1)
      {
        // prepare everything for face evaluation between actual_cluster and actual_cluster+1
        t1 = t1sl;
        t2 = t2sl;

        if (std::abs(t2-t1)>relative_tolerance*fastest_time_step) // only do this if necessary
          {
            // setup the element list that need evaluation (all of actual_cluster who have slower neighbor
            // and all of actual_cluster+1 who have faster neighbor)
            for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
              {
                if (cell_cluster_ids[e]==actual_cluster && cell_have_slower_neighbor[e] && update_cell[e])
                  evaluate_cell[e] = true;
                else if (cell_cluster_ids[e]==actual_cluster+1 && cell_have_faster_neighbor[e] && is_neighbor_of_update_cell[e])
                  evaluate_cell[e] = true;
                else
                  evaluate_cell[e] = false;
              }

            // setup the face list that need evaluation (if face is in between actual cluster and slower cluster)
            // also, setup the masks for the write functions
            for (unsigned int f=0; f<op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches(); ++f)
              {
                evaluate_face[f] = false;
                phi_to_dst[f] = std::bitset<Operator::n_vect>(false);
                phi_neighbor_to_dst[f] = std::bitset<Operator::n_vect>(false);
                phi_to_fluxmemory[f] = std::bitset<Operator::n_vect>(false);
                phi_neighbor_to_fluxmemory[f] = std::bitset<Operator::n_vect>(false);
                // inner faces
                if (f<op.get_matrix_free().n_inner_face_batches())
                  {
                    for (unsigned int v=0; v<Operator::n_vect; ++v)
                      if (mf_faceinfo_cellsminus[v][f]!=numbers::invalid_unsigned_int && mf_faceinfo_cellsplus[v][f]!=numbers::invalid_unsigned_int)
                        {
                          if (evaluate_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect] && evaluate_cell[mf_faceinfo_cellsplus[v][f]/Operator::n_vect])
                            {
                              // set face evaluation
                              if (  ( cell_cluster_ids[mf_faceinfo_cellsminus[v][f]/Operator::n_vect]==actual_cluster
                                      && cell_cluster_ids[mf_faceinfo_cellsplus[v][f]/Operator::n_vect]!=actual_cluster
                                      && is_neighbor_of_update_cell[mf_faceinfo_cellsplus[v][f]/Operator::n_vect] )
                                    ||
                                    ( cell_cluster_ids[mf_faceinfo_cellsminus[v][f]/Operator::n_vect]!=actual_cluster
                                      && cell_cluster_ids[mf_faceinfo_cellsplus[v][f]/Operator::n_vect]==actual_cluster
                                      && is_neighbor_of_update_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect] ) )
                                {
                                  evaluate_face[f] = true;
                                  // set masks
                                  if (cell_cluster_ids[mf_faceinfo_cellsminus[v][f]/Operator::n_vect]==actual_cluster)
                                    {
                                      phi_to_dst[f][v] = true;
                                      if (write_to_fluxmemory)
                                        phi_neighbor_to_fluxmemory[f][v] = true;
                                    }
                                  else
                                    {
                                      if (write_to_fluxmemory)
                                        phi_to_fluxmemory[f][v] = true;
                                      phi_neighbor_to_dst[f][v] = true;
                                    }
                                }
                            }
                        }
                  }
                else // boundary faces
                  {
                    // no contribution from boundary faces in this stage
                  }
              }
            // do first ader
            op.evaluate_cells_and_faces_first_ader(local_state,dst);
          }
      } // check the slower cluster

    // contribution from the same cluster
    {
      t1 = t1sa;
      t2 = t2sa;

      // setup the element list that need evaluation (all of actual_cluster)
      for (unsigned int e=0; e<n_cells_with_ghosts; ++e)
        {
          if (cell_cluster_ids[e]==actual_cluster && update_cell[e])
            evaluate_cell[e] = true;
          else if (cell_cluster_ids[e]==actual_cluster && is_neighbor_of_update_cell[e])
            evaluate_cell[e] = true;
          else
            evaluate_cell[e] = false;
        }

      // setup the face list that need evaluation (if face is in between two cells with actual cluster)
      // also, setup the masks for the write functions
      for (unsigned int f=0; f<op.get_matrix_free().n_inner_face_batches()+op.get_matrix_free().n_boundary_face_batches(); ++f)
        {
          evaluate_face[f] = false;
          phi_to_dst[f] = std::bitset<Operator::n_vect>(false);
          phi_neighbor_to_dst[f] = std::bitset<Operator::n_vect>(false);
          phi_to_fluxmemory[f] = std::bitset<Operator::n_vect>(false);
          phi_neighbor_to_fluxmemory[f] = std::bitset<Operator::n_vect>(false);
          // inner faces
          if (f<op.get_matrix_free().n_inner_face_batches())
            {
              for (unsigned int v=0; v<Operator::n_vect; ++v)
                if (mf_faceinfo_cellsminus[v][f]!=numbers::invalid_unsigned_int && mf_faceinfo_cellsplus[v][f]!=numbers::invalid_unsigned_int)
                  {
                    if ( (update_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect] && update_cell[mf_faceinfo_cellsplus[v][f]/Operator::n_vect])
                         || (update_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect] && evaluate_cell[mf_faceinfo_cellsplus[v][f]/Operator::n_vect] && cell_cluster_ids[mf_faceinfo_cellsplus[v][f]/Operator::n_vect] == actual_cluster)
                         || (evaluate_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect] && update_cell[mf_faceinfo_cellsplus[v][f]/Operator::n_vect] && cell_cluster_ids[mf_faceinfo_cellsminus[v][f]/Operator::n_vect] == actual_cluster) )
                      {
                        evaluate_face[f] = true;
                        // set masks
                        phi_to_dst[f][v] = true;
                        phi_neighbor_to_dst[f][v] = true;
                      }
                  }
            }
          else // boundary faces
            {
              for (unsigned int v=0; v<Operator::n_vect; ++v)
                {
                  if (mf_faceinfo_cellsminus[v][f]!=numbers::invalid_unsigned_int)
                    if (evaluate_cell[mf_faceinfo_cellsminus[v][f]/Operator::n_vect])
                      {
                        evaluate_face[f] = true;
                        phi_to_dst[f][v] = true;
                      }
                }
            }
        }
      // do first ader
      op.evaluate_cells_and_faces_first_ader(local_state,dst);
    }
    // sum the flux memory contribution from all processors
    {
      op.communicate_flux_memory();
    }
    // do the update
    {
      // the evaluate cell vector can be reused
      // only the time step size is needed
      op.evaluate_cells_second_ader(local_state,dst);
    }
    local_state -= dst;
    dst = 0;
  }

}

#endif
