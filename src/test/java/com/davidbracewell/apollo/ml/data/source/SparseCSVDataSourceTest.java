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

package com.davidbracewell.apollo.ml.data.source;

import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.io.Resources;
import org.junit.Test;

import java.util.Map;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class SparseCSVDataSourceTest {
   @Test
   public void stream() throws Exception {
      String data = "label:true,a:1,b:2\n" +
                       "label:true,c:1,b:2\n" +
                       "label:false,c:3\n" +
                       "label:false,c:4,a:2";

      SparseCSVDataSource csv = new SparseCSVDataSource(Resources.fromString(data), "label");
      assertEquals(4, csv.stream().count());

      Map<Object, Long> labelCounts = csv.stream().flatMap(Instance::getLabelSpace).countByValue();
      assertEquals(2, labelCounts.get("true").longValue());
      assertEquals(2, labelCounts.get("false").longValue());

      Map<String, Long> featureCounts = csv.stream().flatMap(Instance::getFeatureSpace).countByValue();
      assertEquals(2, featureCounts.get("a").longValue());
      assertEquals(2, featureCounts.get("b").longValue());
      assertEquals(3, featureCounts.get("c").longValue());
   }

}