/*
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

package com.gengoai.apollo.ml.model;

import com.gengoai.Validation;
import com.gengoai.conversion.Cast;
import com.gengoai.io.Compression;
import com.gengoai.io.resource.Resource;
import lombok.NonNull;

import java.io.IOException;

/**
 * <p>
 * Methods for saving and loading models.
 * </p>
 *
 * @author David B. Bracewell
 */
public final class ModelIO {

   /**
    * Loads a model from the given resource
    *
    * @param resource the resource containing the model
    * @return the model
    * @throws IOException Something went wrong reading the resource
    */
   public static Model load(@NonNull Resource resource) throws IOException {
      Object o = resource.readObject();
      Validation.checkArgumentIsInstanceOf(o, Model.class);
      return Cast.as(o);
   }

   /**
    * Saves a model to the given resource
    *
    * @param model    the model to save
    * @param resource the resource where to write the model
    * @throws IOException Something went wrong writting the resource
    */
   public static void save(@NonNull Model model, @NonNull Resource resource) throws IOException {
      resource.setCompression(Compression.GZIP).writeObject(model);
   }

   private ModelIO() {
      throw new IllegalAccessError();
   }

}//END OF ModelIO
