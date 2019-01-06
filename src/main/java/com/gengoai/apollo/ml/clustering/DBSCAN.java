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
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.ModelParameters;
import com.gengoai.apollo.ml.data.Dataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.conversion.Cast;
import com.gengoai.function.SerializableSupplier;
import com.gengoai.stream.MStream;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;

import java.util.List;

/**
 * <p>
 * A wrapper around the DBSCAN clustering algorithm in Apache Math.
 * </p>
 *
 * @author David B. Bracewell
 */
public class DBSCAN extends Clusterer {

   /**
    * Instantiates a new Dbscan.
    *
    * @param preprocessors the preprocessors
    */
   public DBSCAN(Preprocessor... preprocessors) {
      super(ModelParameters.indexedLabelVectorizer().preprocessors(preprocessors));
   }

   /**
    * Instantiates a new Dbscan.
    *
    * @param modelParameters the model parameters
    */
   public DBSCAN(ModelParameters modelParameters) {
      super(modelParameters);
   }

   /**
    * Clusters the given NDArray using DBSCAN fit parameters
    *
    * @param dataSupplier  the data supplier
    * @param fitParameters the fit parameters
    * @return the flat clustering
    */
   public FlatClustering fit(SerializableSupplier<MStream<NDArray>> dataSupplier, Parameters fitParameters) {
      DBSCANClusterer<ApacheClusterable> clusterer = new DBSCANClusterer<>(fitParameters.eps,
                                                                           fitParameters.minPts,
                                                                           new ApacheDistanceMeasure(
                                                                              fitParameters.measure));
      FlatClustering centroids = new FlatClustering(fitParameters.measure);
      List<ApacheClusterable> apacheClusterables = dataSupplier.get()
                                                               .parallel()
                                                               .map(ApacheClusterable::new)
                                                               .collect();

      List<org.apache.commons.math3.ml.clustering.Cluster<ApacheClusterable>> result = clusterer.cluster(
         apacheClusterables);
      for (int i = 0; i < result.size(); i++) {
         Cluster cp = new Cluster();
         cp.setId(i);
         cp.setCentroid(result.get(i).getPoints().get(0).getVector());
         centroids.add(cp);
      }

      apacheClusterables.forEach(a -> {
         NDArray n = a.getVector();
         int index = -1;
         double score = fitParameters.measure.getOptimum().startingValue();
         for (int i = 0; i < centroids.size(); i++) {
            Cluster c = centroids.get(i);
            double s = fitParameters.measure.calculate(n, c.getCentroid());
            if (fitParameters.measure.getOptimum().test(s, score)) {
               index = i;
               score = s;
            }
         }
         centroids.get(index).addPoint(n);
      });
      return centroids;
   }

   @Override
   public Clustering fitPreprocessed(Dataset dataSupplier, FitParameters fitParameters) {
      return fit(() -> dataSupplier.stream().map(this::encode), Cast.as(fitParameters, Parameters.class));
   }

   @Override
   public Parameters getDefaultFitParameters() {
      return new Parameters();
   }

   /**
    * FitParameters for DBSCAN
    */
   public static class Parameters extends ClusterParameters {
      /**
       * the maximum distance between two vectors to be in the same region
       */
      public double eps = 1.0;
      /**
       * the minimum number of points to form  a dense region
       */
      public int minPts = 2;
   }

}//END OF DBSCAN
