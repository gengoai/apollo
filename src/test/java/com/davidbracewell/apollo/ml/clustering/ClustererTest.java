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

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.data.Dataset;
import com.davidbracewell.apollo.ml.data.DatasetType;
import com.davidbracewell.io.CSV;
import com.davidbracewell.io.CSVReader;
import com.davidbracewell.io.Resources;
import lombok.SneakyThrows;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

/**
 * @author David B. Bracewell
 */
public abstract class ClustererTest {
  private Clusterer algorithm;

  public ClustererTest(Clusterer algorithm) {
    this.algorithm = algorithm;
  }

  protected boolean isDistributed() {
    return false;
  }

  @SneakyThrows
  public Dataset<Instance> getData() {
    List<Instance> instances = new ArrayList<>();
    try (CSVReader reader = CSV.builder()
                               .reader(Resources.fromClasspath("com/davidbracewell/apollo/ml/clustering/sample.csv"))) {
      reader.stream().forEach(row ->
                                instances.add(
                                  Instance.create(
                                    Arrays.asList(Feature.real("X", Double.valueOf(row.get(1))),
                                                  Feature.real("Y", Double.valueOf(row.get(2)))
                                                 ),
                                    row.get(0)
                                                 )
                                             )
                             );
    }
    return Dataset.classification()
                  .type(isDistributed() ? DatasetType.Distributed : DatasetType.InMemory)
                  .featureEncoder(new IndexEncoder())
                  .localSource(instances.stream())
                  .build();
  }

  public Clustering cluster() {
    return this.algorithm.train(getData());
  }

}
