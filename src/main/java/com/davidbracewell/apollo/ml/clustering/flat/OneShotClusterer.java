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

package com.davidbracewell.apollo.ml.clustering.flat;


import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterable;
import com.davidbracewell.apollo.similarity.DistanceMeasure;
import com.google.common.base.Preconditions;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public class OneShotClusterer<T extends Clusterable> implements FlatClusterer<T>, Serializable {

  private static final long serialVersionUID = 1L;
  private DistanceMeasure distanceMeasure;
  private double threshold;

  public OneShotClusterer(double threshold, DistanceMeasure measure) {
    this.threshold = threshold;
    this.distanceMeasure = Preconditions.checkNotNull(measure);
  }

  @Override
  public FlatClustering<T> cluster(List<? extends T> instances) {
    if (instances == null || instances.isEmpty()) {
      return new FlatClustering<>(Collections.<Cluster<T>>emptyList());
    }
    List<Cluster<T>> clusters = new ArrayList<>();

    for (T ii : instances) {
      double minD = Double.POSITIVE_INFINITY;
      int minI = 0;
      for (int k = 1; k < clusters.size(); k++) {
        double d = distance(ii.getPoint(), clusters.get(k));
        if (d < minD) {
          minD = d;
          minI = k;
        }
      }

      if (minD <= threshold) {
        clusters.get(minI).addPoint(ii);
      } else {
        Cluster<T> newCluster = new Cluster<T>();
        newCluster.addPoint(ii);
        clusters.add(newCluster);
      }

    }


    return new FlatClustering<>(clusters);
  }


  private double distance(Vector ii, Cluster<T> cluster) {
    double d = 0;
    for (T jj : cluster) {
      d += distanceMeasure.calculate(ii, jj.getPoint());
    }
    return d / (double) cluster.size();
  }

  public DistanceMeasure getDistanceMeasure() {
    return distanceMeasure;
  }

  public void setDistanceMeasure(DistanceMeasure distanceMeasure) {
    this.distanceMeasure = distanceMeasure;
  }

  public double getThreshold() {
    return threshold;
  }

  public void setThreshold(double threshold) {
    this.threshold = threshold;
  }
}//END OF OneShotClusterer
