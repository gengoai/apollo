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

package com.davidbracewell.apollo.ml.data;

import com.davidbracewell.apollo.ml.IndexEncoder;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.apollo.ml.LabelIndexEncoder;
import com.davidbracewell.apollo.ml.preprocess.PreprocessorList;
import com.davidbracewell.stream.MStream;
import com.davidbracewell.stream.StreamingContext;
import org.junit.Before;

/**
 * @author David B. Bracewell
 */
public class MStreamDatasetTest extends DatasetTest {
   @Before
   public void setUp() throws Exception {
      MStream<Instance> stream = StreamingContext.local().stream(createInstances());
      dataset = new MStreamDataset<>(new IndexEncoder(),
                                     new LabelIndexEncoder(),
                                     PreprocessorList.empty(),
                                     stream);
   }
}//END OF InMemoryDatasetTest
