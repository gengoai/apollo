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

import com.gengoai.ParameterDef;
import com.gengoai.apollo.linear.NDArray;
import com.gengoai.apollo.ml.DiscretePipeline;
import com.gengoai.apollo.ml.FitParameters;
import com.gengoai.apollo.ml.data.VectorizedDataset;
import com.gengoai.apollo.ml.preprocess.Preprocessor;
import com.gengoai.conversion.Cast;
import lombok.NonNull;
import org.apache.commons.math3.ml.clustering.DBSCANClusterer;

import java.util.List;

/**
 * <p>
 * A wrapper around the DBSCAN clustering algorithm in Apache Math. DBSCAN is a flat centroid based clustering where the
 * number of clusters does not need to be specified.
 * </p>
 *
 * @author David B. Bracewell
 */
public class DBSCAN extends FlatCentroidClusterer {
   private static final long serialVersionUID = 1L;
   public static final ParameterDef<Double> eps = ParameterDef.doubleParam("eps");
   public static final ParameterDef<Integer> minPts = ParameterDef.intParam("minPts");

   /**
    * Instantiates a new Dbscan.
    *
    * @param preprocessors the preprocessors
    */
   public DBSCAN(Preprocessor... preprocessors) {
      super(preprocessors);
   }


   /**
    * Instantiates a new Dbscan.
    *
    * @param modelParameters the model parameters
    */
   public DBSCAN(DiscretePipeline modelParameters) {
      super(modelParameters);
   }

   @Override
   public void fit(@NonNull VectorizedDataset vectorStream, @NonNull FitParameters<?> parameters) {
      Parameters fitParameters = Cast.as(parameters);
      setMeasure(fitParameters.measure.value());
      DBSCANClusterer<ApacheClusterable> clusterer = new DBSCANClusterer<>(fitParameters.eps.value(),
                                                                           fitParameters.minPts.value(),
                                                                           new ApacheDistanceMeasure(
                                                                                 fitParameters.measure.value()));
      List<ApacheClusterable> apacheClusterables = vectorStream.parallelStream()
                                                               .map(ApacheClusterable::new)
                                                               .collect();

      List<org.apache.commons.math3.ml.clustering.Cluster<ApacheClusterable>> result = clusterer.cluster(
            apacheClusterables);
      for(int i = 0; i < result.size(); i++) {
         Cluster cp = new Cluster();
         cp.setId(i);
         cp.setCentroid(result.get(i).getPoints().get(0).getVector());
         add(cp);
      }

      apacheClusterables.forEach(a -> {
         NDArray n = a.getVector();
         int index = -1;
         double score = fitParameters.measure.value().getOptimum().startingValue();
         for(int i = 0; i < size(); i++) {
            Cluster c = get(i);
            double s = fitParameters.measure.value().calculate(n, c.getCentroid());
            if(fitParameters.measure.value().getOptimum().test(s, score)) {
               index = i;
               score = s;
            }
         }
         get(index).addPoint(n);
      });
   }

   @Override
   public Parameters getFitParameters() {
      return new Parameters();
   }

   /**
    * FitParameters for DBSCAN
    */
   public static class Parameters extends ClusterParameters<Parameters> {
      /**
       * the maximum distance between two vectors to be in the same region
       */
      public final Parameter<Double> eps = parameter(DBSCAN.eps, 1.0);
      /**
       * the minimum number of points to form  a dense region
       */
      public final Parameter<Integer> minPts = parameter(DBSCAN.minPts, 2);
   }

}//END OF DBSCAN
