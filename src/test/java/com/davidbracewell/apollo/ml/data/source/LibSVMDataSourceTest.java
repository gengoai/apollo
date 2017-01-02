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

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class LibSVMDataSourceTest {
   @Test
   public void binary() throws Exception {
      LibSVMDataSource svm = new LibSVMDataSource(Resources.fromString("+1 1:1 2:3"));

      assertEquals(1, svm.stream().count());
      assertEquals(1, svm.stream().flatMap(Instance::getLabelSpace).countByValue().get("true").longValue());

      Instance ii = svm.stream().first().orElse(null);
      assertNotNull(ii);

      assertEquals(1.0, ii.getValue("0"), 0.0);
      assertEquals(3.0, ii.getValue("1"), 0.0);
   }


   @Test
   public void multiclass() throws Exception {
      LibSVMDataSource svm = new LibSVMDataSource(Resources.fromString("lion 1:1 2:3"),true);

      assertEquals(1, svm.stream().count());
      assertEquals(1, svm.stream().flatMap(Instance::getLabelSpace).countByValue().get("lion").longValue());

      Instance ii = svm.stream().first().orElse(null);
      assertNotNull(ii);

      assertEquals(1.0, ii.getValue("0"), 0.0);
      assertEquals(3.0, ii.getValue("1"), 0.0);
   }

}