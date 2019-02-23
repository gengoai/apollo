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
 *
 */

package com.gengoai.apollo.ml.clustering;

import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.linear.NDArrayFactory;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.apollo.statistics.measure.Distance;
import com.gengoai.conversion.Cast;
import com.gengoai.stream.MStream;
import org.apache.spark.mllib.clustering.KMeansModel;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;

import static com.gengoai.tuple.Tuples.$;

/**
 * <p>Distributed version of KMeans using Apache Spark.</p>
 *
 * @author David B. Bracewell
 */
public class DistributedKMeans extends FlatCentroidClusterer {
   private static final long serialVersionUID = 1L;

   /**
    * Instantiates a new Distributed k means.
    *
    * @param preprocessors the preprocessors
    */
   public DistributedKMeans(Preprocessor... preprocessors) {
      super(preprocessors);
   }

   /**
    * Instantiates a new Distributed k means.
    *
    * @param modelParameters the model parameters
    */
   public DistributedKMeans(DiscretePipeline modelParameters) {
      super(modelParameters);
   }

   @Override
   public void fit(MStream<NDArray> dataSupplier, FitParameters parameters) {
      Parameters fitParameters = Cast.as(parameters);
      org.apache.spark.mllib.clustering.KMeans kMeans = new org.apache.spark.mllib.clustering.KMeans();
      kMeans.setK(fitParameters.K);
      kMeans.setMaxIterations(fitParameters.maxIterations);
      kMeans.setEpsilon(fitParameters.tolerance);
      setMeasure(Distance.Euclidean);
      KMeansModel model = kMeans.run(dataSupplier
                                        .toDistributedStream()
                                        .getRDD()
                                        .map(n -> (Vector) new DenseVector(n.toDoubleArray()))
                                        .cache()
                                        .rdd());

      for (int i = 0; i < model.clusterCenters().length; i++) {
         Cluster cluster = new Cluster();
         cluster.setId(i);
         cluster.setCentroid(NDArrayFactory.rowVector(model.clusterCenters()[i].toArray()));
         add(cluster);
      }

      dataSupplier.map(n -> $(model.predict(new DenseVector(n.toDoubleArray())), n))
                  .forEachLocal(t -> get(t.v1).addPoint(t.v2));

      for (Cluster cluster : this) {
         cluster.setScore(cluster.getScore() / cluster.size());
      }
   }


   @Override
   public DistributedKMeans.Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   /**
    * Fit Parameters for KMeans
    */
   public static class Parameters extends ClusterParameters {
      /**
       * The number of clusters
       */
      public int K = 2;
      /**
       * The maximum number of iterations to run the clusterer for
       */
      public int maxIterations = 100;
      /**
       * The tolerance in change of in-group variance for determining if k-means has converged
       */
      public double tolerance = 1e-3;

   }
}//END OF DistributedKMeans
