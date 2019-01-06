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
import com.gengoai.apollo.ml.Feature;
import com.gengoai.apollo.ml.Instance;
import com.gengoai.io.resource.Resource;
import com.gengoai.stream.MStream;
import com.gengoai.stream.StreamingContext;
import com.gengoai.string.Strings;

import java.io.IOException;
import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

/**
 * <p>Reads in {@link com.gengoai.apollo.ml.data.Dataset} in LibSVM format.</p>
 *
 * @author David B. Bracewell
 */
public class LibSVMDataFormat implements DataFormat, Serializable {
   private static final long serialVersionUID = 1L;
   private final boolean multiClass;
   private final boolean distributed;


   /**
    * Instantiates a new LibSVMDataFormat.
    *
    * @param multiClass  True - treat data as multiclass, False treat as binary
    * @param distributed True - create a distributed stream, False create a local stream
    */
   public LibSVMDataFormat(boolean multiClass, boolean distributed) {
      this.multiClass = multiClass;
      this.distributed = distributed;
   }


   private Example processLine(String line) {
      String[] parts = line.split("\\s+");
      List<Feature> featureList = new ArrayList<>();
      Object target;
      if (multiClass) {
         target = parts[0];
      } else {
         switch (parts[0]) {
            case "+1":
            case "1":
               target = "true";
               break;
            case "-1":
               target = "false";
               break;
            default:
               target = parts[0];
               break;
         }
      }
      for (int j = 1; j < parts.length; j++) {
         String[] data = parts[j].split(":");
         int fnum = Integer.parseInt(data[0]) - 1;
         double val = Double.parseDouble(data[1]);
         featureList.add(Feature.realFeature(Integer.toString(fnum), val));
      }
      return new Instance(target, featureList);
   }

   @Override
   public MStream<Example> read(Resource location) throws IOException {
      return StreamingContext.get(distributed)
                             .textFile(location)
                             .filter(Strings::isNotNullOrBlank)
                             .map(this::processLine);
   }


}//END OF LibSVMDataFormat
