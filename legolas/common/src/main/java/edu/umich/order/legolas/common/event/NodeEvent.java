/*
 *  @author Haoze Wu <haoze@jhu.edu>
 *
 *  The Legolas Project
 *
 *  Copyright (c) 2024, University of Michigan, EECS, OrderLab.
 *      All rights reserved.
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *  http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */
package edu.umich.order.legolas.common.event;

import edu.umich.order.legolas.common.record.RecordWriter;
import java.io.IOException;

/**
 *
 */
public abstract class NodeEvent extends Event {
    public final int serverId;

    public NodeEvent(long nano, int serverId) {
        super(nano);
        this.serverId = serverId;
    }

    @Override
    public void dump(final RecordWriter writer) throws IOException {
        super.dump(writer);
        writer.append(serverId);
    }
}
