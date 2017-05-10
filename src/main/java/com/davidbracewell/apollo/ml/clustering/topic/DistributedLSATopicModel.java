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

import com.davidbracewell.apollo.affinity.Similarity;
import com.davidbracewell.apollo.linalg.Matrix;
import com.davidbracewell.apollo.ml.clustering.Cluster;
import com.davidbracewell.apollo.ml.clustering.Clusterer;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.SparkStream;
import lombok.Getter;
import lombok.Setter;
import org.apache.spark.mllib.linalg.Vector;
import org.apache.spark.mllib.linalg.distributed.RowMatrix;

import static com.davidbracewell.apollo.linalg.SparkLinearAlgebra.applySVD;
import static com.davidbracewell.apollo.linalg.SparkLinearAlgebra.toMatrix;

/**
 * @author David B. Bracewell
 */
public class DistributedLSATopicModel extends Clusterer<LSAModel> {
    private static final long serialVersionUID = 1L;

    @Getter
    @Setter
    private int minCount = 5;
    @Getter
    @Setter
    private int maxVocab = 100_000;
    @Getter
    @Setter
    private double tolerance = 1e-10;
    @Getter
    @Setter
    private int dimension = 300;
    @Getter
    @Setter
    private double rCond = 1e-9;

    @Override
    public LSAModel cluster(MStream<com.davidbracewell.apollo.linalg.Vector> instances) {
        RowMatrix mat = new RowMatrix(new SparkStream<>(instances)
                                          .map(i -> (Vector) new org.apache.spark.mllib.linalg.DenseVector(i.toArray()))
                                          .cache()
                                          .getRDD()
                                          .rdd());

        LSAModel model = new LSAModel(getEncoderPair(), Similarity.Cosine.asDistanceMeasure());
        Matrix topics = toMatrix(applySVD(mat, dimension, rCond, tolerance).U()).transpose();
        model.dimension = topics.numberOfColumns();
        for (int i = 0; i < topics.numberOfRows(); i++) {
            Cluster c = new Cluster();
            c.addPoint(topics
                           .row(i)
                           .copy());
            model.addCluster(c);
        }

        return model;
    }

    @Override
    public void reset() {

    }

}//END OF SparkLSA
