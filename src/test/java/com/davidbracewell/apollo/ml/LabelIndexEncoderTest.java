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

import com.davidbracewell.stream.StreamingContext;
import org.junit.Before;
import org.junit.Test;

import static junit.framework.TestCase.assertTrue;
import static org.junit.Assert.*;

/**
 * @author David B. Bracewell
 */
public class LabelIndexEncoderTest extends AbstractEncoderTest {

   @Before
   public void setUp() throws Exception {
      encoder = new LabelIndexEncoder();
   }

   @Test
   public void get() throws Exception {
      encoder.fit(dataset);
      assertTrue(encoder.get("true") != -1);
   }

   @Test
   public void index() throws Exception {
      encoder.fit(dataset);
      assertTrue(encoder.index("true") != -1);
   }

   @Test
   public void encode() throws Exception {
      encoder.fit(StreamingContext.local().stream("true"));
      assertTrue(encoder.encode("true") != -1);
   }

   @Test
   public void decode() throws Exception {
      encoder.fit(dataset);
      int id = encoder.index("true");
      assertEquals("true", encoder.decode(id));
   }


}