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

package com.gengoai.apollo.ml.clustering.hierarchical;


import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.apollo.ml.clustering.Cluster;
import com.gengoai.apollo.ml.clustering.Clusterer;
import com.gengoai.apollo.ml.clustering.Clustering;
import com.gengoai.apollo.ml.clustering.flat.FlatCentroidClustering;
import com.gengoai.apollo.stat.measure.Measure;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;

/**
 * <p>A clustering for hierarchical clustering techniques where clusters form a tree.</p>
 *
 * @author David B. Bracewell
 */
public class HierarchicalClustering extends Clustering {
   private static final long serialVersionUID = 1L;
   Cluster root;
   Linkage linkage;

   public HierarchicalClustering(Clusterer<?> clusterer, Measure measure) {
      super(clusterer, measure);
   }

   /**
    * Converts the hierarchical clustering into a flat clustering using the given threshold. Each subtree whose
    * inter-cluster distance is less than the given threshold will be flattened into one cluster.
    *
    * @param threshold the threshold to determine how to flatten clusters
    * @return the flat clustering
    */
   public Clustering asFlat(double threshold) {
      List<Cluster> flat = new ArrayList<>();
      process(root, flat, threshold);
      FlatCentroidClustering kmeans = new FlatCentroidClustering(this);
      flat.forEach(kmeans::addCluster);
      return kmeans;
   }

   @Override
   public Cluster get(int index) {
      if (index == 0) {
         return root;
      }
      throw new IndexOutOfBoundsException();
   }

   @Override
   public Cluster getRoot() {
      return root;
   }

   @Override
   public int hardCluster(@NonNull Instance instance) {
      return 0;
   }

   @Override
   public boolean isHierarchical() {
      return true;
   }

   @Override
   public Iterator<Cluster> iterator() {
      return Collections.singleton(root).iterator();
   }

   private void process(Cluster c, List<Cluster> flat, double threshold) {
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

   @Override
   public int size() {
      return 1;
   }

   @Override
   public double[] softCluster(@NonNull Instance instance) {
      NDArray vector = getPreprocessors().apply(instance).toVector(getEncoderPair());
      return new double[]{
         linkage.calculate(vector, root, getMeasure())
      };
   }


}//END OF HierarchicalClustering
