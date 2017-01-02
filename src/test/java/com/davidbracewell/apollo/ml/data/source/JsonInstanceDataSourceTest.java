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

import com.davidbracewell.apollo.ml.Feature;
import com.davidbracewell.apollo.ml.Instance;
import com.davidbracewell.io.Resources;
import org.junit.Test;

import java.util.Collections;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class JsonInstanceDataSourceTest {
   @Test
   public void stream() throws Exception {
      JsonInstanceDataSource ds = new JsonInstanceDataSource(Resources.fromString(
         Instance.create(Collections.singletonList(Feature.TRUE("test")), "lion").toJson()
                                                                                 ));
      assertEquals(1, ds.stream().count());
      assertEquals(1, ds.stream().flatMap(Instance::getLabelSpace).countByValue().get("lion").longValue());

      Instance ii = ds.stream().first().orElse(null);
      assertNotNull(ii);

      assertEquals(1.0, ii.getValue("test"), 0.0);
   }

}