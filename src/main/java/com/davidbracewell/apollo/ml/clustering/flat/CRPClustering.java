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
import com.davidbracewell.apollo.affinity.Optimum;
import com.davidbracewell.apollo.linalg.Vector;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.FeatureVector;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.stream.StreamingContext;
import com.davidbracewell.tuple.Tuple2;
import lombok.NonNull;

import java.util.Arrays;

import static com.davidbracewell.tuple.Tuples.$;

/**
 * @author David B. Bracewell
 */
class CRPClustering extends FlatClustering {
  private static final long serialVersionUID = 1L;

  CRPClustering(@NonNull EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
    super(encoderPair, distanceMeasure);
  }

  @Override
  public int hardCluster(@NonNull Instance instance) {
    return Optimum.MINIMUM.selectBest(softCluster(instance)).v1;
  }

  @Override
  public double[] softCluster(Instance instance) {
    double[] distances = new double[size()];
    Arrays.fill(distances, Double.POSITIVE_INFINITY);
    FeatureVector vector = instance.toVector(getEncoderPair());
    Tuple2<Integer, Double> best = StreamingContext.local().stream(this)
      .parallel()
      .map(cluster -> {
          double max = 0;
          for (Vector jj : cluster) {
            max = Math.max(max, getDistanceMeasure().calculate(vector, jj));
          }
          return $(cluster.getId(), max);
        }
      ).min((t1, t2) -> Double.compare(t1.v2, t2.v2))
      .orElse($(-1, 0.0));

    if (best.v1 != -1) {
      distances[best.v1] = best.v2;
    }
    return distances;
  }

}//END OF CRPClustering
