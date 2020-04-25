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

package com.gengoai.apollo.ml.observation;

import com.gengoai.json.Json;
import com.gengoai.json.JsonEntry;
import com.gengoai.json.JsonMarshaller;

import java.lang.reflect.Type;

/**
 * @author David B. Bracewell
 */
public class VariableCollectionMarshaller extends JsonMarshaller<VariableCollection> {
   @Override
   protected VariableCollection deserialize(JsonEntry entry, Type type) {
      VariableList vl = new VariableList();
      entry.getProperty("variables")
           .elementIterator()
           .forEachRemaining(e -> vl.add(e.getAs(Variable.class)));
      return vl;
   }

   @Override
   protected JsonEntry serialize(VariableCollection variables, Type type) {
      JsonEntry object = JsonEntry.object();
      object.addProperty(Json.CLASS_NAME_PROPERTY, VariableCollection.class.getName());
      JsonEntry array = JsonEntry.array();
      for(Variable variable : variables) {
         array.addValue(variable);
      }
      object.addProperty("variables", array);
      return object;
   }
}//END OF VariableCollectionMarshaller
