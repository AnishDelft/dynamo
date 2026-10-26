// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use async_trait::async_trait;
use dynamo_global_common::Lease;

use super::PoolDiscoveryError;
use super::record::PoolRecord;

/// Reply to `register`. A PoolRelay registers again every heartbeat
/// interval, and the record expires a lease duration after the last
/// registration the store saw, measured on the store's clock. It does not
/// list router replicas: Relays get those from a [`super::ReplicaDirectory`],
/// served as `GET /routers`.
#[derive(Clone, Debug)]
pub struct Registration {
    pub lease: Lease,
}

/// Write side behind `POST /pools`. A PoolRelay calls `register` on a timer,
/// and each call is also the pool's heartbeat: registering an unchanged
/// record again only renews it. Both stores keep records in this process;
/// the MCS store joins them with what MCS imports.
///
/// V1 assumes a trusted network, where any caller may register any pool, so
/// `register` does not take the caller.
#[async_trait]
pub trait PoolRegistrar: Send + Sync {
    async fn register(&self, record: PoolRecord) -> Result<Registration, PoolDiscoveryError>;
}

/// Lets service instances share registrations (see `crate::peers`): each
/// shares the ones its own PoolRelays made, and keeps copies of the others'.
pub trait RegistrationExchange: Send + Sync {
    /// The live registrations PoolRelays made with this instance, with the
    /// time each has left. Copies are left out, so each registration is shared
    /// only by the instance that received it.
    fn local_registrations(&self) -> Vec<(PoolRecord, Duration)>;

    /// Keeps a copy of a registration another instance received, for the
    /// `remaining` time it had there.
    fn adopt(&self, record: PoolRecord, remaining: Duration);
}
