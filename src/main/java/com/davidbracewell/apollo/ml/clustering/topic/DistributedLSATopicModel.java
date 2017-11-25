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

package com.davidbracewell.apollo.ml.clustering.topic;

import com.davidbracewell.apollo.linear.Axis;
import com.davidbracewell.apollo.linear.NDArray;
import com.davidbracewell.apollo.linear.NDArrayFactory;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.apollo.stat.measure.Similarity;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.SparkStream;
import lombok.Getter;
import lombok.Setter;
import org.apache.spark.mllib.linalg.DenseVector;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

import static com.davidbracewell.apollo.linear.SparkLinearAlgebra.sparkSVD;
import static com.davidbracewell.apollo.linear.SparkLinearAlgebra.toMatrix;

/**
 * @author David B. Bracewell
 */
public class DistributedLSATopicModel extends Clusterer<LSAModel> {
   private static final long serialVersionUID = 1L;
   @Getter
   @Setter
   private int K = 100;

   @Override
   public LSAModel cluster(MStream<NDArray> instances) {
      //Create document x word matrix
      SparkStream<Vector> stream = new SparkStream<>(instances.map(i -> (Vector) new DenseVector(i.toArray()))).cache();
      RowMatrix mat = new RowMatrix(stream.getRDD().rdd());

      //since we have document x word, V is the word x component matrix
      // U = document x component, E = singular components, V = word x component
      // Transpose V to get component (topics) x words
      NDArray topics = toMatrix(sparkSVD(mat, K).V().transpose());
      LSAModel model = new LSAModel(this, Similarity.Cosine.asDistanceMeasure(), K);
      for (int i = 0; i < K; i++) {
         Cluster c = new Cluster();
         c.addPoint(NDArrayFactory.wrap(topics.getVector(i, Axis.ROW).toArray()));
         model.addCluster(c);
      }
      return model;
   }

   @Override
   public void resetLearnerParameters() {

   }

}//END OF SparkLSA
