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


import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.linalg.LabeledVector;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.apollo.ml.clustering.Clustering;
import com.google.common.base.Preconditions;
import lombok.NonNull;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

/**
 * The type One shot clusterer.
 *
 * @author David B. Bracewell
 */
public class OneShotClusterer extends Clusterer {
  private static final long serialVersionUID = 1L;
  private DistanceMeasure distanceMeasure;
  private double threshold;

  /**
   * Instantiates a new One shot clusterer.
   *
   * @param threshold the threshold
   * @param measure   the measure
   */
  public OneShotClusterer(double threshold, DistanceMeasure measure) {
    this.threshold = threshold;
    this.distanceMeasure = Preconditions.checkNotNull(measure);
  }

  @Override
  public Clustering cluster(@NonNull List<LabeledVector> instances) {
    OneShotClustering clustering = new OneShotClustering(getEncoderPair(), distanceMeasure);
    clustering.clusters = new ArrayList<>();

    for (LabeledVector ii : instances) {
      double minD = Double.POSITIVE_INFINITY;
      int minI = 0;
      for (int k = 0; k < clustering.clusters.size(); k++) {
        double d = distance(ii, clustering.clusters.get(k));
        if (d < minD) {
          minD = d;
          minI = k;
        }
      }

      if (minD <= threshold) {
        clustering.clusters.get(minI).addPoint(ii);
      } else {
        Cluster newCluster = new Cluster();
        newCluster.addPoint(ii);
        clustering.clusters.add(newCluster);
      }

    }

    for (Iterator<Cluster> itr = clustering.clusters.iterator(); itr.hasNext(); ) {
      Cluster c = itr.next();
      if (c == null || c.size() == 0) {
        itr.remove();
      }
    }

    return clustering;
  }


  private double distance(Vector ii, Cluster cluster) {
    double d = 0;
    for (LabeledVector jj : cluster) {
      d += distanceMeasure.calculate(ii, jj);
    }
    return d / (double) cluster.size();
  }

  /**
   * Gets distance measure.
   *
   * @return the distance measure
   */
  public DistanceMeasure getDistanceMeasure() {
    return distanceMeasure;
  }

  /**
   * Sets distance measure.
   *
   * @param distanceMeasure the distance measure
   */
  public void setDistanceMeasure(DistanceMeasure distanceMeasure) {
    this.distanceMeasure = distanceMeasure;
  }

  /**
   * Gets threshold.
   *
   * @return the threshold
   */
  public double getThreshold() {
    return threshold;
  }

  /**
   * Sets threshold.
   *
   * @param threshold the threshold
   */
  public void setThreshold(double threshold) {
    this.threshold = threshold;
  }


}//END OF OneShotClusterer
