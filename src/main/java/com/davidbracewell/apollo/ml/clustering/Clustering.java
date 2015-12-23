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

package com.davidbracewell.apollo.ml.clustering;

import com.davidbracewell.apollo.ApolloMath;
import com.davidbracewell.apollo.affinity.DistanceMeasure;
import com.davidbracewell.apollo.ml.EncoderPair;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.Model;
import lombok.NonNull;

import java.util.List;


/**
 * @author David B. Bracewell
 */
public abstract class Clustering extends Model {
  private static final long serialVersionUID = 1L;
  private final DistanceMeasure distanceMeasure;

  protected Clustering(EncoderPair encoderPair, DistanceMeasure distanceMeasure) {
    super(encoderPair);
    this.distanceMeasure = distanceMeasure;
  }

  public DistanceMeasure getDistanceMeasure() {
    return distanceMeasure;
  }

  public abstract int size();

  public abstract Cluster get(int index);

  public boolean isFlat() {
    return false;
  }

  public boolean isHierarchical() {
    return false;
  }

  public abstract Cluster getRoot();

  public abstract List<Cluster> getClusters();

  public int hardCluster(@NonNull Instance instance) {
    return ApolloMath.argMin(softCluster(instance)).getV1();
  }

  public abstract double[] softCluster(Instance instance);


}//END OF Clustering
