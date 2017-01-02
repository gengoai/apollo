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

package com.davidbracewell.apollo.ml;

import org.junit.Test;

import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class FeatureTest {

   @Test
   public void componentTest() throws Exception {
      Feature f1 = Feature.TRUE("WORD", "praise");
      assertEquals("WORD=praise", f1.getName());
      assertEquals("WORD", f1.getPrefix());
      assertEquals("praise", f1.getPredicate());
   }

   @Test
   public void trueTest() throws Exception {
      Feature f1 = Feature.TRUE("WORD[-1]=praise");
      assertEquals("WORD[-1]=praise", f1.getName());
      assertEquals("WORD", f1.getPrefix());
      assertEquals("praise", f1.getPredicate());
   }


}