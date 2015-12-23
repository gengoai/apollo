/*
 * (c) 2005 David B. Bracewell
 *
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

package com.davidbracewell.apollo.ml.clustering.hierarchical;


import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterable;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import com.davidbracewell.apollo.ml.clustering.flat.FlatClustering;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class HierarchicalClustering<T extends Clusterable> implements Clustering<T>, Serializable {
  private static final long serialVersionUID = 1L;

  private final Cluster<T> root;
  private final List<Cluster<T>> clusters;

  public HierarchicalClustering(List<Cluster<T>> clusters, Cluster<T> root) {
    this.clusters = clusters;
    this.root = root;
  }

  public Cluster<T> getRoot() {
    return root;
  }

  @Override
  public Iterator<Cluster<T>> iterator() {
    return clusters.iterator();
  }

  @Override
  public int size() {
    return clusters.size();
  }

  public FlatClustering<T> asFlat(double threshold) {
    List<Cluster<T>> flat = new ArrayList<>();
    process(root, flat, threshold);
    return new FlatClustering<>(flat);
  }

  private void process(Cluster<T> c, List<Cluster<T>> flat, double threshold) {
    if (c == null) {
      return;
    }
    if (c.getScore() <= threshold) {
      flat.add(c);
    } else {
      process(c.getLeft(), flat, threshold);
      process(c.getRight(), flat, threshold);
    }
  }


}//END OF HierarchicalClustering
