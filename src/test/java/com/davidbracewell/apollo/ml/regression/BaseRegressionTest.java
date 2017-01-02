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

package com.davidbracewell.apollo.ml.regression;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import org.junit.Test;

import java.util.Arrays;

import static com.davidbracewell.collection.map.Maps.asMap;
import static com.davidbracewell.tuple.Tuples.$;
import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public abstract class BaseRegressionTest {

   private final RegressionLearner learner;
   private final double targetMSE;
   private final double tolerance;

   public BaseRegressionTest(RegressionLearner learner, double targetMSE, double tolerance) {
      this.learner = learner;
      this.targetMSE = targetMSE;
      this.tolerance = tolerance;
   }

   public Dataset<Instance> getData() {
      return Dataset.regression()
                    .source(Arrays.asList(Instance.create(asMap($("A", 10.0), $("B", 5.0)), 100),
                                          Instance.create(asMap($("A", 20.0), $("B", 15.0)), 100),
                                          Instance.create(asMap($("A", 100.0), $("B", 1.0)), 50),
                                          Instance.create(asMap($("A", 120.0), $("B", 0.5)), 50),
                                          Instance.create(asMap($("A", 15.0), $("B", 5.0)), 100),
                                          Instance.create(asMap($("A", 20.0), $("B", 15.0)), 100),
                                          Instance.create(asMap($("A", 120.0), $("B", 1.0)), 50),
                                          Instance.create(asMap($("A", 120.0), $("B", 0.5)), 50)
                                         ));
   }


   @Test
   public void testClassification() throws Exception {
      Regression r = learner.train(getData());
      RegressionEvaluation re = new RegressionEvaluation();
      re.evaluate(r, getData());
      assertEquals(targetMSE, re.meanSquaredError(), tolerance);
   }


}