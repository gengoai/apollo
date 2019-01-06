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
 *
 */

package com.gengoai.apollo.ml.data.format;

import com.gengoai.apollo.ml.Example;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;

import java.io.IOException;

/**
 * <p>
 * A data format provides a methodology for reading in examples from a given resource in a specific format and creating
 * a stream of {@link Example}s.
 * </p>
 *
 * @author David B. Bracewell
 */
public interface DataFormat {

   /**
    * Reads and converts the data in the given resource into a stream of examples.
    *
    * @param location the location of the data to read
    * @return the stream of examples
    * @throws IOException Something went wrong reading the data.
    */
   MStream<Example> read(Resource location) throws IOException;

}//END OF DataSource
