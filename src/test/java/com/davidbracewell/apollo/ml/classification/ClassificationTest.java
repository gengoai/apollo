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

package com.davidbracewell.apollo.ml.classification;

import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.source.DenseCSVDataSource;
import com.davidbracewell.io.Resources;
import lombok.SneakyThrows;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class ClassificationTest {

   private final ClassifierLearner learner;
   private final double targetAcc;
   private final double tolerance;

   protected ClassificationTest(ClassifierLearner learner, double targetAcc, double tolerance) {
      this.learner = learner;
      this.targetAcc = targetAcc;
      this.tolerance = tolerance;
   }

   @SneakyThrows
   public Dataset<Instance> getData() {
      return Dataset.classification().dataSource(
         new DenseCSVDataSource(Resources.fromClasspath("com/davidbracewell/apollo/ml/test.csv"),
                                true))
                    .featureEncoder(new IndexEncoder())
                    .build();
   }

   @Test
   public void testClassification() throws Exception {
      Dataset<Instance> data = getData();
      Classifier classifier = learner.train(data);
      ClassifierEvaluation evaluation = new ClassifierEvaluation();
      evaluation.evaluate(classifier, data);
      assertEquals(targetAcc, evaluation.accuracy(), tolerance);
      System.out.println(learner.getClass().getSimpleName() + "\t" + evaluation.accuracy());
   }


}//END OF ClassificationTest
